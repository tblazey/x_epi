"""
Utility functions for XSeq
"""

# Load libraries
import os
import re
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

# Default locations for resource files
BASE_DIR = os.path.dirname(__file__)
RES_DIR = os.path.join(BASE_DIR, "res")

# Available nuclei
NUCLEI = {
    "13C": 10.7084e6,
    "1H": 42.57638474e6,
    "2H": 6.536e6,
    "15N": -4.316e6,
    "17O": -5.722e6,
    "31P": 17.235e6,
    "19F": 40.078e6,
    "23Na": 11.262e6,
    "129Xe": -11.777e6,
}


def nuc_to_gamma(nuc):
    """
    Returns gyromagnetic ratio given nucleus

    Parameters
    ----------
    nuc : str
       Nucleus string (e.g., 13C, 1H, etc)

    Returns
    -------
    gamma : float
       Gyromangetic ratio in Hz/T
    """

    if nuc in NUCLEI:
        return NUCLEI[nuc]
    raise KeyError("Unknown nucleus")


def load_ssrf_grad(grd_path):
    """
    Load in SSRF gradient

    Parameters
    ----------
    grd_path : str
       Path to SSRF gradient file

    Returns
    -------
    grd_data : ndarray
       Numpy array containing gradient data
    grd_max : float
       Maximum gradient value (G/cm)
    grd_delta : float
       Sampling time (s)
    """

    # Open file to get maximum gradient strength and time resolution
    with open(grd_path, encoding="utf-8") as fid:
        grd_txt = fid.read()
    try:
        grd_max = float(re.search("Max Gradient Strength = .*", grd_txt)[0].split()[4])
        grd_delta = float(re.search("Resolution = .*", grd_txt)[0].split()[2]) * 1e-6
    except TypeError as e:
        raise TypeError(
            "Cannot find max gradient strength and time resolution "
            f"information in {grd_path}."
        ) from e

    # Load in gradient data. A bit inefficient, but more readable
    grd_data = np.loadtxt(grd_path, usecols=[0], comments="#")

    return grd_data, grd_max, grd_delta


def load_ssrf_rf(rf_path):
    """
    Load in SSRF rf

    Parameters
    ----------
    rf_path : str
       Path to SSRF RF file

    Returns
    -------
    rf_mag : ndarray
       Numpy array containing rf magnitude
    rf_pha : ndarray
       Numpy array containing rf phase
    b1_max : float
       Maximum B1 (G)
    rf_delta : float
       Sampling time (s)
    """

    # Open file to get maximum b1 and time resolution
    with open(rf_path, encoding="utf-8") as fid:
        rf_txt = fid.read()
    try:
        b1_max = float(re.search("Max B1 = .*", rf_txt)[0].split()[3])
        rf_delta = float(re.search("Resolution = .*", rf_txt)[0].split()[2]) * 1e-6
    except TypeError as e:
        raise TypeError(f"Cannot find max B1 and time resolution in {rf_path}.") from e

    # Load in rf data. A bit inefficient, but more readable
    rf_data = np.loadtxt(rf_path, usecols=[0, 1], comments="#")

    return rf_data[:, 1], rf_data[:, 0], b1_max, rf_delta


def interp_waveform(sig, delta_t, delta_ti, ti_end=None):
    """
    Interpolates signal waveform

    Parameters
    ----------
    sig : ndarray
       Array containing waveform data
    delta_t : float
       Sampling time for data in sig
    delta_ti : float
       Sampling time to interpolate to
    ti_end : float
       Time to end interpolation at

    Returns
    -------
    sig_i : ndarray
       Array containing interpolated data
    """

    # Get time vectors
    n = sig.shape[0]
    t = np.arange(1, n + 1) * delta_t
    if ti_end is None:
        ti_end = np.ceil(t[-1] / delta_ti) * delta_ti
    ti = np.arange(delta_ti, ti_end + delta_ti, delta_ti)

    # Interpolate
    return interp.interp1d(t, sig, bounds_error=False, fill_value=0)(ti)


def compute_k_space(seq):
    """
    Computes rf, gradient, and k-space waveforms of current sequence

    Parameters
    ----------
    seq : XSeq object
       Sequence object containing waveforms

    Returns
    -------
    g_list : list
       Gradient waveforms in X, Y, and Z dimensions for each metabolite
    rf_list : list
       Complex rf waveforms for each metabolite
    k_list : list
       3D k-space waveforms for each metabolite. Returned if k_3d is True
    adc_list : list
       Timings for 3D k-space waveforms for each metabolite. Returned if k_3d is True
    k_2d_list : list
       2D k-space waveforms for each metabolite. Returned if k_2d is True
    adc_2d_list : list
       Timings for 2D k-space waveforms for each metabolite. Returned if k_2d is True
    """

    # Get waveforms
    [wave_data, _, _, t_adc, _] = seq.waveforms_and_times(append_RF=True)

    # Resample gradient times so that they are all the same
    t_max = [wave_data[0][0, -1], wave_data[1][0, -1]]
    if wave_data[2].shape[1] > 0:
        t_max.append(wave_data[2][0, -1])
    t_max = np.max(t_max)
    t_i = np.arange(0, t_max + seq.system.grad_raster_time, seq.system.grad_raster_time)
    x_i = interp.interp1d(
        wave_data[0][0, :], wave_data[0][1, :], fill_value=0.0, bounds_error=False
    )(t_i)
    y_i = interp.interp1d(
        wave_data[1][0, :], wave_data[1][1, :], fill_value=0.0, bounds_error=False
    )(t_i)
    if wave_data[2].shape[1] > 0:
        z_i = interp.interp1d(
            wave_data[2][0, :], wave_data[2][1, :], fill_value=0.0, bounds_error=False
        )(t_i)
    else:
        z_i = np.zeros_like(t_i)
    rf_i = interp.interp1d(
        np.real(wave_data[3][0, :]),
        wave_data[3][1, :],
        fill_value=0.0,
        bounds_error=False,
    )(t_i)

    # Find right edges of rf/blocks
    rf_edges = [[] for i in range(seq.n_met)]
    rf_durs = [[] for i in range(seq.n_met)]
    block_edges = np.cumsum(seq.block_durations)
    for i in range(1, len(seq.block_events) + 1):
        rf_num = seq.block_events[i][1]
        met_idx = seq.blck_lbls[i - 1]
        if rf_num > 0 and isinstance(met_idx, int):
            rf_edges[met_idx].append(block_edges[i - 1])
            rf_durs[met_idx].append(seq.block_durations[i - 1])

    # Make empty arrays for storing data
    k_list = []  # k-space points in 3d plane
    k_2d_list = []  # k-space points in 2d plane
    adc_list = []  # adc times in 3d plane
    adc_2d_list = []  # adc times in 2d plane
    g_list = []  # interpolated gradients
    rf_list = []  # interpolated rf waveforms

    # Loop through metabolites
    for m in range(seq.n_met):
        # Compute k-space, restart integration at each rf edge
        k_x = np.zeros(t_i.shape[0])
        k_y = np.zeros_like(k_x)
        k_z = np.zeros_like(k_x)
        k_mask = np.zeros(k_x.shape[0], dtype=bool)
        for idx, edge in enumerate(rf_edges[m]):
            # Different logic for last rf
            if idx == len(rf_edges[m]) - 1:
                # Get edge that is common for last rf for all metabolites
                edge_time = (
                    np.round(edge / seq.system.grad_raster_time)
                    * seq.system.grad_raster_time
                )
                mask = t_i >= edge_time

                # Edges that very depending on if we are doing last metabolite or not
                if m == seq.n_met - 1 and idx == 0:
                    grad_mask = t_i > (edge - rf_durs[m][0])
                if m != seq.n_met - 1:
                    edge_time_2 = rf_edges[m + 1][0] - rf_durs[m + 1][0]
                    edge_time_2 = (
                        np.round(edge_time_2 / seq.system.grad_raster_time)
                        * seq.system.grad_raster_time
                    )
                    mask = np.logical_and(mask, t_i <= edge_time_2)
                    if idx == 0:
                        grad_mask = np.logical_and(
                            t_i > (edge - rf_durs[m][0]),
                            t_i <= (rf_edges[m + 1][0] - rf_durs[m + 1][0]),
                        )
            else:
                # Mask for all times before end of block
                end_time = rf_edges[m][idx + 1] - rf_durs[m][idx + 1]
                end_time = (
                    np.round(end_time / seq.system.grad_raster_time)
                    * seq.system.grad_raster_time
                )
                end_mask = t_i <= end_time

                # Mask for gradients
                if idx == 0:
                    start_time = edge - rf_durs[m][idx]
                    start_time = (
                        np.round(start_time / seq.system.grad_raster_time)
                        * seq.system.grad_raster_time
                    )
                    start_mask = t_i >= start_time
                    grad_mask = np.logical_and(start_mask, end_mask)

                # Get mask for curent block
                edge_time = (
                    np.round(edge / seq.system.grad_raster_time)
                    * seq.system.grad_raster_time
                )
                edge_mask = t_i >= edge_time
                mask = np.logical_and(edge_mask, end_mask)

            # Compute k-space
            k_x[mask] = integ.cumulative_trapezoid(x_i[mask], t_i[mask], initial=0.0)
            k_y[mask] = integ.cumulative_trapezoid(y_i[mask], t_i[mask], initial=0.0)
            k_z[mask] = integ.cumulative_trapezoid(z_i[mask], t_i[mask], initial=0.0)
            if idx == 0:
                k_mask_2d = np.copy(mask)
            k_mask = np.logical_or(k_mask, mask)

        # Figure out k-space locations for each adc sample time
        t_mask = np.logical_and(
            t_adc >= np.min(t_i[k_mask]), t_adc <= np.max(t_i[k_mask])
        )
        t_mask_2d = np.logical_and(
            t_adc[t_mask] >= np.min(t_i[k_mask_2d]),
            t_adc[t_mask] <= np.max(t_i[k_mask_2d]),
        )
        adc_x = interp.interp1d(t_i[k_mask], k_x[k_mask])(t_adc[t_mask])
        adc_y = interp.interp1d(t_i[k_mask], k_y[k_mask])(t_adc[t_mask])
        adc_z = interp.interp1d(t_i[k_mask], k_z[k_mask])(t_adc[t_mask])
        k_list.append(np.stack((k_x[k_mask], k_y[k_mask], k_z[k_mask])))
        k_2d_list.append(np.stack((k_x[k_mask_2d], k_y[k_mask_2d])))
        adc_list.append(np.stack((adc_x, adc_y, adc_z)))
        adc_2d_list.append(np.stack((adc_x[t_mask_2d], adc_y[t_mask_2d])))
        g_list.append(np.stack((x_i[grad_mask], y_i[grad_mask], z_i[grad_mask])))
        rf_list.append(rf_i[grad_mask])

    # Return waveforms
    return k_list, adc_list, k_2d_list, adc_2d_list, g_list, rf_list


def save_k_space(seq, out):
    """
    Parameters
    ----------
    seq : XSeq sequence object
       Sequence object to compute k-space data
    out : str
       Root for output files

    Returns
    -------
    k_space : list of ndarrays
       List containing k_space data for each metabolite
    """

    # Get k-space data
    _, k_adc, _, _, _, _ = compute_k_space(seq)

    # Save it as a numpy file
    k_arr = np.empty(len(k_adc), dtype=object)
    k_arr[:] = k_adc
    np.save(out + "_kspace.npy", k_arr)


def r_spline_basis(x, knots, norm=False):
    """
    Calculates a restricted cubic spline basis for X given a set of knots

    Parameters
    ----------
    x : array
       A array of length n containing the x-values for cubic spline basis
    knots: array
       An array of length p containing knot locations
    norm: logical
       If true normalizes spline (see rcspline.eval norm=2 in R)

    Returns
    -------
    basis : matrix
       an n  xp basis for a restricted cubic spine
    """

    # Check number of knots
    n_k = knots.shape[0]
    # if n_k <= 2:
    #   raise RuntimeError('Number of knots must be at least 3')

    # Get normalization factor
    if norm is True:
        norm_fac = np.power(knots[-1] - knots[0], 2)
    else:
        norm_fac = 1

    # Create array to store basis matrix
    n_p = x.shape[0]
    basis = np.ones((n_p, n_k))

    # Set second basis function to x-value
    basis[:, 1] = x

    # Loop through free knots
    for i in range(n_k - 2):
        # First part of basis function
        t_one = np.maximum(0, np.power(x - knots[i], 3))

        # Second part of basis function
        scale_d = knots[n_k - 1] - knots[n_k - 2]
        scale_two = (knots[n_k - 1] - knots[i]) / scale_d
        t_two = np.maximum(0, np.power(x - knots[n_k - 2], 3)) * scale_two

        # You get the drill
        scale_three = (knots[n_k - 2] - knots[i]) / scale_d
        t_three = np.maximum(0, np.power(x - knots[n_k - 1], 3)) * scale_three

        # Compute the basis function.
        basis[:, i + 2] = (t_one - t_two + t_three) / norm_fac

    return basis


def knot_loc(x, n_k, bounds=None):
    """
    Calculates location for knots based on sample quantiles

    Parameters
    ----------
    x : array
       A array of length n containing the x-values
    n_k: interger
      Number of knots
    bounds: array
       A 2 x 1 array containing percentile bounds.
       If not set then function uses method described below.

    Returns
    -------
    knots : array
       A set of knot locations

    Notes
    -----
    Uses the same basic algorithm as Hmisc package in R:
       For 3 knots -> outer percentiles are 10 and 90%
       For 4-6 knots -> outer percentiels are 5% and 95%
       For >6 knots -> outer percentiles are 2.5% and 97.5%
       All other knots are linearly spaced between outer percentiles
    """

    # Set boundary knot percentiles
    # if n_k <= 2:
    #   raise RuntimeError('Number of knots must be at least 3')
    if bounds is not None:
        b_knots = [bounds[0], bounds[1]]
    elif 2 >= n_k <= 3:
        b_knots = [10, 90]
    elif 4 >= n_k <= 6:
        b_knots = [5, 95]
    elif 6 > n_k <= x.shape[0]:
        b_knots = [2.5, 97.5]
    else:
        raise RuntimeError(f'Cannot determine knot locations for {n_k} knots')

    # Get percentiles for all knots
    knot_per = np.linspace(b_knots[0], b_knots[1], n_k)

    # Get actual knot locations based upon percentiles
    knots = np.percentile(x, knot_per)

    return knots
