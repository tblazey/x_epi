"""
x_epi data class module
"""

# Load libraries
from itertools import product
import json
import types
from importlib.metadata import version
import nibabel as nib
import numpy as np
from numpy.fft import ifftn, ifftshift, fft, fftn, fftshift, fftfreq
from pynufft import NUFFT
from scipy.interpolate import interp1d
import twixtools
from .utils import r_spline_basis, knot_loc


class XData:
    """
    Data class for processing data created with an x_epi sequence
    """

    def __init__(
        self,
        fov=[240, 240, 240],
        rbw=50,
        n_avg=1,
        n_rep=1,
        ts=1000,
        ro_off=0,
        pe_off=0,
        slc_off=0,
        n_echo=1,
        delta_te=0,
        symm_ro=True,
        acq_3d=True,
        ramp_samp=False,
        alt_read=False,
        alt_pha=False,
        alt_slc=False,
        ori="Transverse",
        pe_dir="AP",
        **kwargs,
    ):
        """
        Initialize X EPI data object

        Attributes
        ----------
        fov : array_like
           Contains FOV of each dimension (mm)
        rbw : float
           Readout bandwidth (kHz)
        n_avg : int
           Number of averages
        n_rep : int
           Number of repeats
        ts : float
           Minimum durartion of each metabolite set (ms)
        ro_off : float
           Spatial offset in readout direction (mm)
        pe_off : float
           Spatial offset in phase encoding direction (mm). Only for recon.
        slc_off: float
           Spatial offset in slice direction (mm)
        n_echo : int
           Number of echoes following each excitation
        delta_te : float
           Minimum duration between each echo (ms)
           #Extract sequence options
        symm_ro : bool
           Uses symmetric readout if True. Otherwise uses flyback.
        acq_3d : bool
            Uses a 3D readout if True. Otherwise uses 2D.
        alt_read : bool
           Alternate polarity of readout every repetition
        alt_pha : bool
           Alternate polarity of phase encoding every repetition
        alt_slc : bool
           Alternate polarity of second phase encoding every repetition
        ori : str
           Image orientation for reconstruction.
        pe_dir : str
           Phase encoding direction for reconstruction

        Returns
        -------
        XData object
        """

        # Primary x_epi specific sequence options
        self.fov = np.array(fov)
        self.rbw = rbw * 1e3  # Hz
        self.n_avg = n_avg
        self.n_rep = n_rep
        self.ts = ts / 1e3  # s
        self.ro_off = ro_off
        self.pe_off = pe_off
        self.slc_off = slc_off
        self.n_echo = n_echo
        self.delta_te = delta_te / 1e3  # s
        self.n_met = 0
        self.mets = []
        self.symm_ro = symm_ro  # True=symmetric, False=flyback
        self.acq_3d = acq_3d  # True = 3D, False = 2D
        self.ramp_samp = ramp_samp  # True = ramp sampling
        self.alt_read = alt_read  # True = Alt ro pol. between reps
        self.alt_pha = alt_pha  # True = Alt pe pol. between reps
        self.alt_slc = alt_slc  # True = Alt pe2 pol. between reps
        self.ori = ori
        self.pe_dir = pe_dir
        self.alt_signs = np.array([alt_read, alt_pha, alt_slc]) * -2 + 1

        if acq_3d is False:
            self.fft_axes = [0, 1]
        else:
            self.fft_axes = [0, 1, 2]

    def add_met(
        self,
        name=None,
        size=[16, 16, 16],
        size_acq=[16, 16, 16],
        pf_pe=1,
        pf_pe2=1,
        freq_off=0,
        z_centric=False,
        esp=0.48e-3,
        k_acq=None,
        k_coord=None,
        k_data=None,
        img_data=None,
        **kwargs,
    ):
        """
        Add metabolite acquisition to sequence

        Parameters
        ----------
        name : str
           Name for metabolite
        size : array_like
           Grid size in x, y, and z
        size_acq :  array_like
            Acquired dimensions in x, y, z
        pf_pe : float
           Partial Fourier fraction in y dimension
        pf_pe2 : float
           Partial Fourier fraction in z dimension
        freq_off : float
           Frequency offset (Hz) of pulse
        z_centric : bool
           Use centric phase encoding in z direction
        esp : float
            Echo spacing (s)
        k_acq : ndarray
            k-space data with dimensions `size_acq`, `n_rep`, `n_echo`
        k_data : ndarrary
            k-space data with dimensions `size`, `n_rep`, `n_echo`
        img_data : ndarray
            image data with dimensions `size`, `n_rep`, `n_echo`
        """

        # Figure out names
        self.n_met += 1
        if name is None:
            name = f"met_{self.n_met}"

        # Add metabolite specific options
        met_obj = types.SimpleNamespace()
        met_obj.name = name
        met_obj.size = size
        met_obj.pf_pe = pf_pe
        met_obj.pf_pe2 = pf_pe2
        met_obj.freq_off = freq_off
        met_obj.z_centric = z_centric
        met_obj.esp = esp
        met_obj.size_acq = size_acq
        met_obj.pe_start = size[1] - size_acq[1]
        met_obj.pe_2_start = size[2] - size_acq[2]
        met_obj.dims = size + [self.n_rep] + [self.n_echo]
        met_obj.dims_acq = size_acq + [self.n_rep] + [self.n_echo]
        met_obj.vox_dims = np.array(self.fov) / np.array(size)
        met_obj.k_acq = k_acq
        met_obj.k_data = k_data
        met_obj.k_coord = k_coord
        met_obj.img_data = img_data
        met_obj.xfm = None

        self.mets.append(met_obj)

    def load_k_coords(self, coord_path):
        """
        Loads in k-space coordinates and adds them to each metabolite object

        Parameters
        ----------
        coord_path : str
            Path to k-space coordinates written by x_epi apps
        """

        # Load in k-space coordinates
        k_coord = np.load(coord_path, allow_pickle=True)

        # Reshape coordinates for each metabolite
        for idx, met in enumerate(self.mets):
            coord_dims = (
                3,
                met.dims_acq[0],
                met.dims_acq[1],
                met.dims_acq[4],
                met.dims_acq[2],
                met.dims_acq[3],
            )
            k_coord[idx] = (
                k_coord[idx]
                .reshape(coord_dims, order="F")
                .transpose([0, 1, 2, 4, 5, 3])
            )
            met.k_coord = k_coord[idx]

    def load_k_data(self, twix_path, recon_dims=True):
        """
        Reads k-space data for each metabolite from Siemens twix file

        Parameters
        ----------
        twix_path : str
            Path to twix data file
        recon_dims : bool
            If true, saves k-space data with same dimensions as eventual reconstructed
            image to `k_data`. Zero-padding is used to fill any data skipped due to
            partial Fourier. If False, k-space data will have dimensions matching
            acquisition and be saved to `k_acq`.
        """

        # Read in twix data
        twix_data = twixtools.read_twix(twix_path, verbose=False)

        # Make empty arrays for storing data
        for met in self.mets:
            if recon_dims is True:
                met.k_data = np.zeros(met.dims, dtype=np.complex128)
            else:
                met.k_acq = np.zeros(met.dims_acq, dtype=np.complex128)

        # Extract data from twix into acquisition matrix
        line_idx = 0
        iter_prod_out = product(range(self.n_avg), range(self.n_rep), self.mets)
        for avg, rep, met in iter_prod_out:
            iter_prod_in = product(
                range(met.size_acq[2]),
                range(self.n_echo),
                range(met.size_acq[1]),
            )
            for pe_2, echo, pe_1 in iter_prod_in:
                k_line = twix_data[-1]["mdb"][line_idx].data.squeeze()
                if recon_dims is True:
                    pe_1_idx = met.pe_start + pe_1
                    pe_2_idx = met.pe_2_start + pe_2
                    met.k_data[:, pe_1_idx, pe_2_idx, rep, echo] += k_line
                else:
                    met.k_acq[:, pe_1, pe_2, rep, echo] += k_line

                line_idx += 1

    def regrid_k_data(self, method='cubic', nufft_size=6):
        """
        Regrid k-space in readout direction so that we have even spacing with the
        correct FOV
        
        Parameters
        ----------
        method : str
            Cubic uses cubic spline interpolation, nufft performs a non-uniform NFFT
        nufft_size : int
            Size of nufft interpolator
        """

        for met in self.mets:
            # Get coordinates to resample to
            n_x = met.size[0]
            l_x = self.fov[0] / 1e3  # m
            x_coord = fftshift(fftfreq(n_x, l_x / n_x))[:, np.newaxis] + 1 / l_x / 2

            # Expand coordinates to accomodate time flipping
            k_coord = met.k_coord
            k_coord = np.repeat(k_coord, 2, axis=4)
            k_coord[:, :, :, :, 1, :] = k_coord[
                :,
                :: self.alt_signs[0],
                :: self.alt_signs[1],
                :: self.alt_signs[2],
                1,
                :,
            ]
            x_coord = np.repeat(x_coord, 2, axis=1)
            x_coord[:, 1] = x_coord[:: self.alt_signs[0], 1]

            # Make empty array for storing interpolated data
            met.k_data = np.zeros(met.dims, dtype=np.complex128)

            # Interpolate oversampled lines to desired resolution
            iter_prod = product(*[range(i) for i in met.dims_acq[1:]])
            for pe_1, pe_2, rep, echo in iter_prod:
                alt_idx = rep % 2
                k_x = k_coord[0, :, pe_1, pe_2, alt_idx, echo]
                k_y = met.k_acq[:, pe_1, pe_2, rep, echo]
                
                # Switch for regridding method
                if method == 'cubic':
                k_line_i = interp1d(
                        k_x, k_y, bounds_error=False, fill_value=0, kind="cubic"
                )(x_coord[:, alt_idx])
                elif method == 'nufft':
                    k_x *= np.pi / np.max(np.abs(k_x))
                    
                    # Setup non-uniform fft
                    nufft = NUFFT()
                    nufft.plan(
                        k_x[:, np.newaxis],
                        (met.size[0], ),
                        (met.size_acq[0], ),
                        (nufft_size, )
                    )
                    k_line_i = fft(fftshift(nufft.solve(k_y, 'cg', maxiter=100)))

                    # Flip if necessary
                    if pe_1 % 2 == 1:
                        k_line_i = k_line_i[::-1]

                # Add data to complete k-space matrix
                pe_1_idx = met.pe_start + pe_1
                pe_2_idx = met.pe_2_start + pe_2
                met.k_data[:, pe_1_idx, pe_2_idx, rep, echo] = k_line_i

    def flip_k_data(self):
        """
        Performs axis flips common to x_epi data
        """

        for met in self.mets:
            # Flip readout on every other line
            if self.symm_ro is True and self.ramp_samp is False:
                met.k_data[:, 1::2] = met.k_data[::-1, 1::2]

            # Flip y on even echoes
            met.k_data[:, :, :, :, 0::2] = met.k_data[:, ::-1, :, :, 0::2]

            # Flip odd time points if necessary.
            if self.n_rep > 1:
                met.k_data[:, :, :, 1::2, :] = met.k_data[
                    :: self.alt_signs[0],
                    :: self.alt_signs[1],
                    :: self.alt_signs[2],
                    1::2,
                    :,
                ]

    def fft_recon(
        self, ref_data=None, slice_idx=None, point=True, n_k=6, fft_axes=[0, 1, 2]
    ):
        """
        Reconstructs metabolite data using the FFT. Also can apply a phase correction
        for symmetrically acquired EPI data

        Parameters
        ----------
        ref_data : XData object
            XData object containing with reference scan data (no phase encoding) for
            each metabolite
        slice_idx : int
            Zero-based slice index to use for reference scan
        point : bool
            Run a pointwise (True) or spline-based (False) phase correction
        n_k : int
            Number of knots for spline-based phase correction
        """

        # Loop through metabolite dimension
        for idx, met in enumerate(self.mets):
            # Get projection of first dimension of reference scan
            if ref_data is not None:
                ref_proj = fftn(ref_data.mets[idx].k_data, axes=[0])

                # Get phase correction angle
                if point is True:
                    # Do pointwise phase correction
                    if slice_idx is not None:
                        pha = np.exp(
                            -1j * np.angle(ref_proj[:, :, slice_idx : slide_idx + 1])
                        )
                    else:
                        pha = np.exp(-1j * np.angle(ref_proj))

                # Do a spline phase correction
                else:
                    # Sum over slices of first echos at first timepoint, may not be wise for 2D
                    ref_sum = np.sum(ref_proj[:, :, :, 0, 0], axis=2)

                    # Average "positive" reference lines (see Heid, 1997)
                    ref_lines = (ref_sum[:, 0::4] + ref_sum[:, 2::4]) / 2

                    # Get "negative" lines in between ref lines
                    mov_lines = ref_sum[:, 1::4]

                    # Get phase difference after summing up ref/mov pairs
                    pha_diff = fftshift(
                        np.angle(np.sum(ref_lines * np.conj(mov_lines), axis=1))
                    )

                    # Fit spline to phase difference
                    x = np.arange(pha_diff.shape[0])
                    knots = knot_loc(x, n_k)
                    basis = r_spline_basis(x, knots)
                    coef, _, _, _ = np.linalg.lstsq(basis, pha_diff, rcond=None)
                    pha_hat = basis @ coef
                    pha = np.ones(met.k_data.shape, dtype=np.complex128)

                    # Flip phase for odd echos
                    for i in range(pha.shape[-1]):
                        if i % 2 == 0:
                            sign = 1
                        else:
                            sign = -1
                        pha[:, :, :, :, i] = np.exp(1j * fftshift(pha_hat) * sign)[
                            :, np.newaxis, np.newaxis, np.newaxis
                        ]

            else:
                pha = np.ones(list(met.k_data.shape[0:3]) + [1, 1])

            # Apply reference scan and transform to image
            met.img_data = fftshift(
                fftn(fftn(met.k_data, axes=[0]) * pha, axes=self.fft_axes[1::]),
                axes=self.fft_axes,
            )

    def apply_off_res(self, pad=8):
        """
        Shifts metabolites that were acquired off-resonance

        Parameters
        ----------
        pad : int
            Size to zero pad each dimension when apply shifts
        """

        for met in self.mets:
            if met.freq_off != 0:
                # Construct time vector for image acquisition
                ro_t = np.arange(met.img_data.shape[0]) / self.rbw
                pe_t = np.arange(met.img_data.shape[0]) * met.esp
                t = ro_t[:, np.newaxis] + pe_t[np.newaxis, :]
                ec_t = np.arange(met.dims[4]) * (met.esp + t[-1, -1])
                t = t[:, :, np.newaxis] + ec_t[np.newaxis, np.newaxis, :]

                # Flip echos
                t[:, 1::2, :] = t[::-1, 1::2, :]
                t[:, :, 0::2] = t[:, ::-1, 0::2]

                # Shift according to fourier shift theorem
                img_shift = np.exp(
                    -2 * np.pi * 1j * t[:, :, np.newaxis, np.newaxis, :] * met.freq_off
                )

                # Apply off resonance correction to the image.
                shifted = ifftshift(
                    ifftn(met.img_data, axes=[0, 1]) * img_shift,
                    axes=[0, 1],
                )

                # Pad to allow subvoxel shifts
                shifted_pad = self._pad_image(shifted, pads=(pad, pad, 0, 0, 0))
                met.img_data = fftn(shifted_pad, axes=[0, 1])[
                    0::pad, 0::pad, :, :, :
                ]

    def _pad_image(self, img, pads=(8, 8, 8, 0, 0)):
        """
        Apply padding to an image

        Parameters
        ----------
        img : ndarrray
            Five dimensional array containing image data to pad
        pads : list
            List of ints pad scales in each dimension

        Returns
        -------
        img_pad : ndarray
            Padded dimensions with dimensions equal to `img` dimensions * `pad`
        """

        # Figure out number of elements to pad from pad scales
        pad_list = []
        for idx, pad in enumerate(pads):
            if pad > 0:
                i_pad = (img.shape[idx] * pad - img.shape[idx]) // 2
            else:
                i_pad = 0
            pad_list.append([i_pad, i_pad])

        return np.pad(img, pad_list)

    def apply_phase_shift(self, pad=8):
        """
        Shifts images in phase encoding directions if image was acquired with an off
        center field of view.

        Parameters
        ----------
        pad : int
            Size to zero pad each dimension when apply shifts
        """

        if self.acq_3d is True and (self.pe_off != 0 or self.slc_off != 0):
            for met in self.mets:
                # Get coordinate grids in image space (mm)
                x_c = fftfreq(met.size[0], met.vox_dims[0]) + 1.0 / 2.0 / self.fov[0]
                y_c = fftfreq(met.size[1], met.vox_dims[1]) + 1.0 / 2.0 / self.fov[1]
                z_c = fftfreq(met.size[2], met.vox_dims[2]) + 1.0 / 2.0 / self.fov[2]
                _, y_g, z_g = np.meshgrid(x_c, y_c, z_c, indexing="ij")

                # Shift via Fourier domain
                shift = np.zeros(met.dims, dtype=np.complex128)
                if self.pe_off != 0:
                    shift += np.exp(2 * 1j * np.pi * y_g * self.pe_off)[
                        :, :, :, np.newaxis, np.newaxis
                    ]
                if self.slc_off != 0:
                    shift += np.exp(2 * 1j * np.pi * z_g * self.slc_off)[
                        :, :, :, np.newaxis, np.newaxis
                    ]
                shifted = ifftshift(
                    ifftn(met.img_data, axes=[1, 2]) * shift, axes=[1, 2]
                )
                shifted_pad = self._pad_image(shifted, pads=(0, pad, pad, 0, 0))
                met.img_data = fftn(shifted_pad, axes=[1, 2])[:, 0::pad, 0::pad, :, :]

    def create_nii_xfm(self):
        """
        Creates NIfTI format affine transforms for each metabolite. Units are in mm.
        """

        for met in self.mets:
            # Create nifti sform matrix, for now only have options for coronal or transverse

            if self.ori == "Coronal":
                met.xfm = np.array(
                    [
                        [0, -met.vox_dims[1], 0, self.fov[1] / 2 - met.vox_dims[1] / 2],
                        [0, 0, met.vox_dims[2], -self.fov[2] / 2 + met.vox_dims[2] / 2],
                        [met.vox_dims[0], 0, 0, -self.fov[0] / 2 + met.vox_dims[0] / 2],
                        [0, 0, 0, 1],
                    ]
                )
                met.xfm[1, 3] += -self.slc_off
                met.xfm[2, 3] += -self.ro_off
            elif self.ori == "Transverse":
                met.xfm = np.array(
                    [
                        [-met.vox_dims[0], 0, 0, self.fov[0] / 2 - met.vox_dims[0] / 2],
                        [0, -met.vox_dims[1], 0, self.fov[1] / 2 - met.vox_dims[1] / 2],
                        [0, 0, -met.vox_dims[2], self.fov[2] / 2 - met.vox_dims[2] / 2],
                        [0, 0, 0, 1],
                    ]
                )
                met.xfm[2, 3] += -self.slc_off
                met.xfm[1, 3] += -self.ro_off
            else:
                raise ValueError("Sagittal scans are not currently supported")

    def save_nii(self, out_root, cmplx=False, mean=False):
        """
        Saves image data to nifti

        Parameters
        ----------
        out_root : str
            Root for output files. Does not include extension or cmplx/mean suffix.
        cmplx : bool
            Saves data as complex is true
        mean : bool
            Saves temporal mean
        """

        for met in self.mets:
            out_data = met.img_data
            suff = ""
            if mean is True:
                out_data = np.mean(out_data, axis=3)
                suff += "_mean"
            if cmplx is False:
                out_data = np.abs(out_data)
            else:
                suff += "_cmplx"

            # Make sure we have a transformation
            if met.xfm is None:
                self.create_nii_xfm()

            # Make nifti class
            nii = nib.Nifti1Image(out_data, affine=met.xfm)
            nii.set_qform(met.xfm, "scanner")
            nii.header.set_xyzt_units(xyz="mm", t="sec")
            zooms = np.append(met.vox_dims, [1])
            if len(out_data.shape) == 5:
                zooms = np.insert(zooms, 3, self.ts)
            nii.header.set_zooms(zooms)

            # Save image
            nii.to_filename(f"{out_root}_{met.name}{suff}.nii.gz")

    def create_param_dic(self):
        """
        Create a dictionary for sequence parameters

        Returns
        -------
        param_dic : dic
           Dictionary of sequence parameters
        """

        # Generic parameters
        out_dic = {
            "fov": [int(dim) for dim in self.fov],
            "rbw": self.rbw / 1e3,
            "n_avg": self.n_avg,
            "n_rep": self.n_rep,
            "ts": self.ts * 1e3,
            "ro_off": self.ro_off,
            "pe_off": self.pe_off,
            "slc_off": self.slc_off,
            "n_echo": self.n_echo,
            "delta_te": self.delta_te * 1e3,
            "n_met": self.n_met,
            "symm_ro": self.symm_ro,
            "acq_3d": self.acq_3d,
            "ramp_samp": self.ramp_samp,
            "alt_read": self.alt_read,
            "alt_pha": self.alt_pha,
            "alt_slc": self.alt_slc,
            "ori": self.ori,
            "pe_dir": self.pe_dir,
            "alt_signs": [int(sign) for sign in self.alt_signs],
            "fft_axes": self.fft_axes,
            "version": version("x_epi"),
        }

        # Metabolite specific parameter
        out_dic["mets"] = []
        for met in self.mets:
            met_dic = {}
            met_dic["name"] = met.name
            met_dic["size"] = met.size
            met_dic["pf_pe"] = met.pf_pe
            met_dic["pf_pe2"] = met.pf_pe2
            met_dic["freq_off"] = met.freq_off
            met_dic["z_centric"] = met.z_centric
            met_dic["esp"] = met.esp
            met_dic["size_acq"] = [int(dim) for dim in met.size_acq]
            met_dic["pe_start"] = met.pe_start
            met_dic["pe_2_start"] = met.pe_2_start
            met_dic["dims"] = met.dims
            met_dic["dims_acq"] = met.dims_acq
            out_dic["mets"].append(met_dic)

        return out_dic

    def save_param_dic(self, out_root):
        """
        Saves input parameters to json

        Parameters
        ----------
        out_root : str
           Path to write json data to. Does not include extension
        """

        with open(f"{out_root}.json", "w", encoding="utf-8") as fid:
            json.dump(self.create_param_dic(), fid, indent=2)
