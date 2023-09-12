"""
XEpi sequence class module
"""

#Load libraries
from ast import literal_eval
from copy import deepcopy
import json
import re
import types
import numpy as np
import pypulseq as pp
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
import scipy.integrate as integ
from .utils import nuc_to_gamma, load_ssrf_grad, load_ssrf_rf, interp_waveform, RES_DIR

class XEpi(pp.Sequence):
    """
    Sequence class for creating EPI sequences.
    Inherits from the Pypulseq sequence class.
    """

    def __init__(self, fov=(240, 240, 240), rbw=50, n_avg=1, n_rep=1, tr=0,
                 tv=0, ts=0, ro_off=0, pe_off=0, slc_off=0, slice_axis='Z',
                 n_echo=1, delta_te=0, symm_ro=True, acq_3d=True, no_pe=False,
                 grad_spoil=False, ramp_samp=False, no_slc=False, alt_read=False,
                 alt_pha=False, alt_slc=False, ro_os=1, b0=3, nuc='13C', rf_dead_time=100,
                 rf_ringdown_time=30, max_slew=200, max_grad=100, ori='Transverse',
                 pe_dir='AP', **kwargs):
        """
        Initialize X EPI sequence object, inherits from pypulseq Sequence object

        Parameters
        ----------
        fov : array_like
           Contains FOV of each dimension (mm)
        rbw : float
           Readout bandwidth (kHz)
        n_avg : int
           Number of averages
        n_rep : int
           Number of repeats
        tr : float
           Minimum duration of each excitation block (ms)
        tv : float
           Minimum duration of each 3D image (ms)
        ts : float
           Minimum durartion of each metabolite set (ms)
        ro_off : float
           Spatial offset in readout direction (mm)
        pe_off : float
           Spatial offset in phase encoding direction (mm). Only for recon.
        slc_off: float
           Spatial offset in slice direction (mm)
        slice_axis : str
           Axis to play slic gradient on. Takes X, Y, Z
        n_echo : int
           Number of echoes following each excitation
        delta_te : float
           Minimum duration between each echo (ms)
           #Extract sequence options
        symm_ro : bool
           Uses symmetric readout if True. Otherwise uses flyback.
        acq_3d : bool
            Uses a 3D readout if True. Otherwise uses 2D.
        no_pe : bool
           Turns off phase encoding gradients is True.
        grad_spoil : bool
           Turns on gradient spoilers if True
        no_slc : bool
           Turns off slice gradients if True.
        alt_read : bool
           Alternate polarity of readout every repetition
        alt_pha : bool
           Alternate polarity of phase encoding every repetition
        alt_slc : bool
           Alternate polarity of second phase encoding every repetition
        ro_os : float
           Readout oversampling factor
        b0 : float
           Main magnetic field strength (T)
        nuc : str
           Nucleus to image
        rf_dead_time : float
           Dead time for RF channel (us)
        rf_ringdown_time : float
           Ringdown time for RF channel (us)
        max_slew : float
           Maximum slew rate (T/m/s)
        max_grad : float
           Maximum gradient strength (mT/m)
        ori : str
           Image orientation for reconstruction.
        pe_dir : str
           Phase encoding direction for reconstruction

        Returns
        -------
        XEpi sequence object
        """

        #Initialize sequence object
        lims = pp.Opts(max_grad=max_grad, grad_unit="mT/m", max_slew=max_slew,
                       slew_unit="T/m/s", rf_ringdown_time=rf_ringdown_time * 1E-6,
                       rf_dead_time=rf_dead_time * 1E-6, gamma=nuc_to_gamma(nuc),
                       B0=b0, adc_dead_time=20E-6)
        super().__init__(lims)

        #Primary x_epi specific sequence options
        self.nuc = nuc
        self.fov = np.array(fov) / 1E3                  #meters
        self.rbw = rbw                                  #Khz
        self.n_avg = n_avg
        self.n_rep = n_rep
        self.tr = tr / 1E3                              #s
        self.tv = tv / 1E3                              #s
        self.ts = ts / 1E3                              #s
        self.ro_off = ro_off / 1E3                      #m
        self.pe_off = pe_off / 1E3                      #m
        self.slc_off = slc_off / 1E3                    #m
        self.slice_axis = slice_axis.lower()
        self.n_echo = n_echo
        self.delta_te = delta_te / 1E3                  #s
        self.n_met = 0
        self.mets = []
        self.symm_ro = symm_ro                          #True=symmetric, False=flyback
        self.acq_3d = acq_3d                            #True = 3D, False = 2D
        self.no_pe = no_pe                              #True = No phase encoding grads
        self.grad_spoil = grad_spoil                    #True = Gradient spoilers
        self.no_slc = no_slc                            #True = No slice grads
        self.ramp_samp = ramp_samp                      #True = ramp sampling
        self.alt_read = alt_read                        #True = Alt ro pol. between reps
        self.alt_pha = alt_pha                          #True = Alt pe pol. between reps
        self.alt_slc = alt_slc                          #True = Alt pe2 pol. between reps
        self.ro_os = ro_os
        self.ori = ori
        self.pe_dir = pe_dir

        #Compute parameters that don't vary between metabolites
        self.delta_v = self.rbw / 2 * 1E3               #hz
        self.dwell = 1 / self.delta_v / 2               #s
        self.gx_amp = 2 * self.delta_v / self.fov[0]    #Hz / m

        #Create empty members
        self.blck_lbls = []
        self.spec = None

    def add_spec(self, run_spec='END', spec_size=2048, spec_bw=25, spec_flip=2.5,
                 spec_tr=1000, n_spec=1, **kwargs):

        """
        Add spectra acquisition to start and/or end of acquisition

        Parameters
        ----------
        run_spec : str
           When to run spectra. Takes START, END, or BOTH
        spec_size : n
           Number of spectra to acquire
        spec_bw : float
           Spectral bandwidth (kHz)
        spec_flip : float
           Flip angle for sinc pulse (degrees)
        spec_tr : float
           Minimum duration of each excitation block (ms)
        n_spec : int
           Number of spectra acquistions (seperated by at least spec_tr)
        """

        #Add spectra options
        self.spec = types.SimpleNamespace()
        self.spec.run = run_spec
        self.spec.size = spec_size
        self.spec.bw = spec_bw * 1E3                    #Hz
        self.spec.flip = spec_flip / 180 * np.pi        #radians
        self.spec.tr = spec_tr / 1E3                    #s
        self.spec.n = n_spec
        self.add_spec_events()

    def add_spec_events(self):
        """
        Creates rf, adc, and slice gradient events for spectra
        """

        #Make rf event
        self.spec.rf, \
        self.spec.gz, \
        self.spec.gz_r = pp.make_sinc_pulse(flip_angle=self.spec.flip, return_gz=True,
                                            system=self.system, use='excitation',
                                            slice_thickness=self.fov[2])
        self.spec.rf.freq_offset += self.spec.gz.amplitude * (not self.slc_off)
        self.spec.gz.amplitude *= not self.slc_off
        self.spec.gz_r.amplitude *= not self.slc_off

        #Make adc event
        self.spec.adc = pp.make_adc(num_samples=self.spec.size, system=self.system,
                                    delay=self.system.adc_dead_time,
                                    duration=self.spec.size / self.spec.bw)

    def add_met(self, name=None, size=(16, 16, 16), pf_pe=1, pf_pe2=1,
                sinc_frac=0.5, sinc_tbw=4, formula='1', use_sinc=False,
                grd_path=f'{RES_DIR}/siemens_singleband_pyr_3T.GRD',
                rf_path=f'{RES_DIR}/siemens_singleband_pyr_3T.RF',
                flip=90, freq_off=0, sinc_dur=4, z_centric=False, **kwargs):
        """
        Add metabolite acquisition to sequence

        Parameters
        ----------
        name : str
           Name for metabolite
        size : array_like
           Grid size in x, y, and z
        pf_pe : float
           Partial Fourier fraction in y dimension
        pf_pe2 : float
           Partial Fourier fraction in z dimension
        sinc_frac : float
           Center fraction of sinc pulse
        sinc_tbw : float
           Time-bandwidth product of sinc pulse
        grd_path : str
           Path to gradient file from Spectral-Spatial RF Toolbox
        rf_path : str
           'Path to rf file from Spectral-Spatial RF Toolbox
        formula : str
           Formula for scaling SSRF gradient where x is slice thickness in mm
        use_sinc : bool
           Use a sinc pulse instead of an SSRF pulse
        flip : float
           Flip angle (degrees)
        freq_off : float
           Frequency offset (Hz) of pulse
        sinc_dur : float
           Duration of sinc pulse (ms)
        z_centric : bool
           Use centric phase encoding in z direction
        """

        #Figure out names
        self.n_met += 1
        if name is None:
            name = f'met_{self.n_met}'

        #Add metabolite specific options
        met_obj = types.SimpleNamespace()
        met_obj.name = name                             #Metabolite name
        met_obj.size = size                             #Size of metabolite grid
        met_obj.pf_pe = pf_pe                           #Partial Fourier fraction in y dimension
        met_obj.pf_pe2 = pf_pe2                         #Partial Fourier fraction in z dimension
        met_obj.use_sinc = use_sinc                     #Are we using a sinc pulse or an ssrf?
        met_obj.flip = flip / 180 * np.pi               #Flip angle in radians
        met_obj.freq_off = freq_off                     #Pulse frequency offset in Hz
        met_obj.sinc_frac = sinc_frac                   #Center fraction of sinc pulse
        met_obj.sinc_tbw = sinc_tbw                     #Time-bandwidth fraction of sinc pulse
        met_obj.sinc_dur = sinc_dur * 1E-3              #Duration of sinc pulse (s)
        met_obj.grd_path = grd_path                     #Path to SSRF gradient shape
        met_obj.rf_path = rf_path                       #Path to SSRF RF shape
        met_obj.formula = formula                       #SSRF gradient scale formula
        met_obj.b1_max = 0                              #Peak B1 (Gauss)
        met_obj.rf_delta = 0                            #RF sampling time (sec)
        met_obj.grd_max = 0                             #Peak gradient strength (G/cm)
        met_obj.grd_delta = 0                           #Gradient sampling time (sec)
        met_obj.z_centric = z_centric                   #Bool for centric encoding

        #Grid grid size that will be acquired
        met_obj.size_acq = np.int32(np.ceil(np.array(met_obj.size) * \
                                            np.array([1, met_obj.pf_pe, met_obj.pf_pe2])))

        #Construct basic phase encoding blip
        gy_blip = pp.make_trapezoid('y', system=self.system, area= 1 / self.fov[1])
        gy_dur = gy_blip.flat_time + gy_blip.rise_time + gy_blip.fall_time
        met_obj.gy_blip_amp = gy_blip.amplitude

        #Construct readout gradient (in Hz/m)
        t_acq = met_obj.size[0] / self.delta_v / 2
        flat_time = np.ceil(t_acq / self.system.grad_raster_time) * \
                    self.system.grad_raster_time
        if self.ramp_samp is False:
            met_obj.gx = pp.make_trapezoid(channel='x', system=self.system,
                                           flat_time=flat_time, amplitude=self.gx_amp)
            met_obj.gx_amp = met_obj.gx.amplitude
            met_obj.dwell = self.dwell

            #Check blip/readout timing
            if gy_dur > (met_obj.gx.rise_time + met_obj.gx.fall_time) and \
               self.symm_ro is True:
                raise RuntimeError('Phase encoding blip cannot fit in between readout plateaus')

        else:

            #First make gradient with area needed for imaging + half a pe blip on each side
            #At this point, we are assuming maximum slew rate
            gy_blip_area = np.power(gy_dur / 2, 2) * self.system.max_slew
            gx_samp_area = met_obj.size[0] / self.fov[0]
            gx_area = gx_samp_area + gy_blip_area

            #Make gradient object
            try:
                met_obj.gx = pp.make_trapezoid('x', area=gx_area, duration=gy_dur + t_acq,
                                               system=self.system)
            except AssertionError as msg:
                min_dur = float(re.findall(r"\d+\.\d+", msg.args[0])[0]) * 1E-6
                min_dur  = np.ceil(min_dur / self.system.grad_raster_time) * \
                           self.system.grad_raster_time
                t_acq = min_dur - gy_dur
                met_obj.gx = pp.make_trapezoid('x', area=gx_area, duration=min_dur,
                                               system=self.system)

            #However the gradient might not use the maximum slew rate, so we need to fix the
            #area within the imaging portion to account for the slower slew rate
            #This is done by adjusting the amplitude in portion to the desired area
            gx_slew = met_obj.gx.amplitude / met_obj.gx.rise_time
            gx_samp_area_slow = met_obj.gx.area - np.power(gy_dur / 2, 2) * gx_slew
            met_obj.gx.amplitude = met_obj.gx.amplitude / gx_samp_area_slow * gx_samp_area
            ramp_time = met_obj.gx.rise_time + met_obj.gx.fall_time
            met_obj.gx.area = met_obj.gx.amplitude * (met_obj.gx.flat_time + 0.5 * ramp_time)
            met_obj.gx.flat_area = met_obj.gx.amplitude * met_obj.gx.flat_time
            met_obj.gx_amp = met_obj.gx.amplitude

            #Compute new dwell time for ramp sampling
            rs_dwell = 1 / met_obj.gx.amplitude / self.fov[0] / self.ro_os
            met_obj.dwell = np.floor(rs_dwell / self.system.adc_raster_time) * \
                            self.system.adc_raster_time
            met_obj.size_acq[0] = np.floor(t_acq / met_obj.dwell / 4) * 4

        #Excitation thickness
        if self.acq_3d is True:
            met_obj.exc_thk = self.fov[2]
        else:
            met_obj.exc_thk = self.fov[2] / met_obj.size[2]

        #Make flyback readout if needed
        if self.symm_ro is False:
            met_obj.fly = pp.make_trapezoid('x', system=self.system, area=-met_obj.gx.area)


        #Data acquisition event. Delay is constructed so that it includes ramp time as
        #wellas adjustments for the rounding that had to occur to match gradient raster
        #and the fact that ADC samples take place in the middle of trapezoidal gradient
        #times. Extra  time for rounding is divided evenly on each side of the readout
        #gradient. The dwell time can be removed from remove (t_acq - t_dwell) if you
        #want no odd/even offset

        adc_freq_off = met_obj.freq_off + self.ro_off * met_obj.gx_amp
        if self.ramp_samp is False:
            adc_delay = met_obj.gx.rise_time + flat_time / 2 - (t_acq) / 2
            met_obj.adc = pp.make_adc(met_obj.size[0], duration=t_acq,
                                      system=self.system, delay=adc_delay,
                                      freq_offset=adc_freq_off)
        else:
            adc_delay = met_obj.gx.rise_time + met_obj.gx.flat_time / 2 - \
                        met_obj.size_acq[0] / 2 * met_obj.dwell
            adc_delay = np.round(adc_delay / 1E-6) * 1E-6
            met_obj.adc = pp.make_adc(met_obj.size_acq[0],
                                      system=self.system, delay=adc_delay,
                                      freq_offset=adc_freq_off, dwell=met_obj.dwell)

        #Construct x prephasing gradient
        met_obj.gx_pre = pp.make_trapezoid('x', system=self.system,
                                          area=-met_obj.gx.area / 2)

        #Make y prephasing gradient
        gy_pre_area = -(met_obj.size[1] - 1) / 2 / self.fov[1]
        if met_obj.pf_pe != 1:
            gy_pre_area += (met_obj.size[1] - met_obj.size_acq[1]) / self.fov[1]
        if np.abs(gy_pre_area) > 0:
            met_obj.gy_pre = pp.make_trapezoid('y', system=self.system, area=gy_pre_area)
        else:
            met_obj.gy_pre = pp.make_trapezoid('y', system=self.system,
                                               area=-met_obj.gx.area / 2)
            met_obj.gy_pre.amplitude = 0
        met_obj.gy_pre_amp = met_obj.gy_pre.amplitude

        #Create maximum z gradient if necessary
        if met_obj.size[2] > 1 and self.acq_3d is True:
            gz_pre_area = (met_obj.size[2] - 1) / 2 / self.fov[2]
            met_obj.gz_pre = pp.make_trapezoid(self.slice_axis, system=self.system,
                                               area=gz_pre_area)
            met_obj.gz_pre_amp = met_obj.gz_pre.amplitude

        #Split phase encoding gradient if necessary
        if self.symm_ro is True:
            gy_split = pp.split_gradient_at(gy_blip, gy_dur / 2, system=self.system)
            gy_up, gy_dn, _ = pp.align(right=gy_split[0], left=[gy_split[1], met_obj.gx])
            gy_dnup = pp.add_gradients((gy_dn, gy_up), system=self.system)
            met_obj.gy = gy_dnup
            met_obj.gy_up = gy_up
            met_obj.gy_dn = gy_dn
        else:
            met_obj.gy = gy_blip

        #Create spoilers
        if self.grad_spoil is True:
            spoil_area = np.abs(met_obj.gx.area)
            met_obj.x_spoil = pp.make_trapezoid('x', system=self.system, area=spoil_area)
            met_obj.y_spoil = pp.make_trapezoid('y', system=self.system, area=spoil_area)
            if self.acq_3d is True and met_obj.size[2] > 1:
                met_obj.z_spoil = met_obj.gz_pre
                met_obj.z_spoil.amplitude = -met_obj.z_spoil.amplitude
            else:
                met_obj.z_spoil = pp.make_trapezoid(self.slice_axis, system=self.system,
                                                    area=spoil_area)
            met_obj.spoil_amp = met_obj.x_spoil.amplitude

        #Compute echo spacing
        met_obj.esp = met_obj.gx.rise_time + met_obj.gx.fall_time + met_obj.gx.flat_time
        if self.symm_ro is False:
            met_obj.esp += met_obj.fly.rise_time + met_obj.fly.fall_time

        #Compute multipliers for slice/second phase encodes
        if self.acq_3d is True:
            met_obj.z_range = np.linspace(1, -1,
                                          met_obj.size[2])[0:int(met_obj.size_acq[2])][::-1]
            if met_obj.z_centric is True:
                z_sort = np.argsort(np.abs(met_obj.z_range))
                met_obj.z_range = met_obj.z_range[z_sort]
        else:
            met_obj.z_range = np.arange(-np.floor(met_obj.size[2] / 2),
                                        np.ceil(met_obj.size[2] / 2))
        #Add metabolite object to list
        self.mets.append(met_obj)

        #Create ssrf events
        if met_obj.use_sinc is False:
            self.create_ssrf_grad_ev(self.n_met - 1)
            self.create_ssrf_rf_ev(self.n_met - 1)

        #Create sinc events
        else:

            #Gradient baseline sinc pulse
            met_obj.rf, \
            met_obj.rf_gz, \
            met_obj.rf_gz_r = pp.make_sinc_pulse(flip_angle=met_obj.flip,
                                                 return_gz=True,
                                                 system=self.system,
                                                 slice_thickness=met_obj.exc_thk,
                                                 use='excitation',
                                                 time_bw_product=met_obj.sinc_tbw,
                                                 duration=met_obj.sinc_dur,
                                                 center_pos=met_obj.sinc_frac)
            met_obj.rf.freq_offset = met_obj.freq_off + met_obj.rf_gz.amplitude * \
                                     self.slc_off
            met_obj.rf_gz.channel = self.slice_axis
            met_obj.rf_gz_r.channel = self.slice_axis
            met_obj.rf_gz.amplitude *= not self.no_slc
            met_obj.rf_gz_r.amplitude *= not self.no_slc

            #Create frequency offsets for slices
            if self.acq_3d is False:

                #Loop through slice multipliers
                met_obj.rf_freq_offs = []
                for z in met_obj.z_range:

                    #Compute the slice offset
                    slice_offset = met_obj.rf_gz.amplitude * \
                                   (met_obj.exc_thk * (z + 0.5) + self.slc_off)
                    met_obj.rf_freq_offs.append(met_obj.freq_off + slice_offset)

    def create_ssrf_rf_ev(self, met_idx):
        """
        Creates RF event for SSRF pulse

        Parameters
        ----------
        met_idx : int
           Index of metabolite in self.mets
        """

        #Extract the metabolite object we need
        met_obj = self.mets[met_idx]

        #Load in rf data
        mag, pha, met_obj.b1_max, met_obj.rf_delta = load_ssrf_rf(met_obj.rf_path)

        #Convert magnitude to Hz and phase to radians.
        #Factor 2 / np.pi is to make it a fraction of 90 degree pulse
        mag *= met_obj.b1_max * 2 * met_obj.flip / np.pi / np.max(mag) * 1E-4 * \
               self.system.gamma
        pha *= np.pi / 180

        #Interpolate rf data to rf raster time
        rf_end = (mag.shape[0] - 1) * met_obj.rf_delta
        rf_end = np.ceil(rf_end / self.system.grad_raster_time) * \
                 self.system.grad_raster_time
        met_obj.mag = interp_waveform(mag, met_obj.rf_delta, self.system.rf_raster_time,
                                ti_end=rf_end)
        met_obj.pha = interp_waveform(pha, met_obj.rf_delta, self.system.rf_raster_time,
                                      ti_end=rf_end)

        #Construct rf event
        pha_off = 2 * np.pi * met_obj.kz * self.slc_off
        ssrf = met_obj.mag * np.exp(1j * (met_obj.pha + pha_off))
        met_obj.rf = pp.make_arbitrary_rf(ssrf, 2 * np.pi, system=self.system,
                                          norm=False, freq_offset=met_obj.freq_off)

        #Construct delay to make RF event a multiple of block duration if necessary
        rf_dur = met_obj.rf.dead_time + met_obj.rf.ringdown_time + met_obj.rf.shape_dur
        delay_aug =  np.ceil(rf_dur / self.system.block_duration_raster) * \
                     self.system.block_duration_raster
        if delay_aug > 0:
            met_obj.rf_delay = pp.make_delay(delay_aug)

    def create_ssrf_grad_ev(self, met_idx):
        """
        Creates gradient event for SSRF pulse

        Parameters
        ----------
        met_idx : int
           Index of metabolite in self.mets
        """

        #Extract the metabolite object we need
        met_obj = self.mets[met_idx]

        #Load in gradient data
        grd_data, met_obj.grd_max, met_obj.grd_delta = load_ssrf_grad(met_obj.grd_path)

        #Convert gradient data to Hz / m
        grd_data *= met_obj.grd_max / np.max(np.abs(grd_data)) * 1E-2 * self.system.gamma

        #Scale gradient using slice/slab thickness formula
        scale_form = met_obj.formula.replace('x', str(met_obj.exc_thk * 1E3))
        met_obj.slc_scale = literal_eval(scale_form)
        grd_data *= met_obj.slc_scale


        #Interpolate gradient waveform to raster
        grd_data_i = interp_waveform(grd_data * (not self.no_slc), met_obj.grd_delta,
                                     self.system.grad_raster_time)

        #Interpolate k-space vector to rf raster time
        kz = integ.cumtrapz(grd_data, dx=met_obj.grd_delta, initial=0)
        kz_end = (grd_data.shape[0] - 1) * met_obj.grd_delta
        kz_end = np.ceil(kz_end / self.system.grad_raster_time) * \
                 self.system.grad_raster_time
        met_obj.kz = interp_waveform(kz, met_obj.grd_delta, self.system.rf_raster_time,
                                     ti_end=kz_end)

        #Make gradient event
        met_obj.rf_gz = make_arbitrary_grad(self.slice_axis, grd_data_i, system=self.system,
                                            delay=self.system.rf_dead_time)
        met_obj.rf_gz.first = 0
        met_obj.rf_gz.last = 0

    def add_blck_lbl(self, *args, lbl=None):
        """
        Quick function to add events to block and add a label to list

        Parameters
        ----------
        args : SimpleNamespace
           Block structure / items to be added to sequence
        lbl : str
           Label to add to self.blck_lbls
        """

        self.add_block(*args)
        if lbl is not None:
            self.blck_lbls.append(lbl)

    def create_seq(self, return_plot=False, no_reps=False):
        """
        Creates epi sequence

        Parameters
        ----------
        return_plot : bool
           If True, returns a sequence object with no repetitions or averages.
           Useful for plotting sequence
        no_reps : bool
           If True, turns off replications/averages

        Returns
        -------
        plot_seq : X EPI object
           Sequence object with no repetitions or aveages
        """

        #Turn off replications to make things go faster
        if no_reps is True:
            n_rep = 1
            n_avg = 1
        else:
            n_rep = self.n_rep
            n_avg = self.n_avg

        #Average loops
        for a in range(n_avg):

            #Add spectra if necessary
            try:
                if self.spec.run in ('START', 'BOTH'):
                    for i in range(self.spec.n):
                        if i == 0:
                            spec_start = len(self.block_durations)
                        self.add_blck_lbl(self.spec.rf, self.spec.gz, lbl='s')
                        self.add_blck_lbl(self.spec.gz_r, lbl='s')
                        self.add_blck_lbl(self.spec.adc, lbl='s')
                        if i == 0:
                            spec_dur = np.sum(self.block_durations[spec_start::])
                            spec_delay = self.spec.tr - spec_dur
                        if spec_delay > 0:
                            self.add_blck_lbl(pp.make_delay(spec_delay), lbl='s')
            except AttributeError:
                pass

            #Loop through repetitions
            for r in range(n_rep):
                ts_delay = self.ts

                #Loop through metabolites
                for m in range(self.n_met):

                    #Extract metabolite object
                    met = self.mets[m]

                    #Alternate polarity of first gradient each repetition if necessary
                    if r > 0:
                        if self.alt_read is True:
                            if m == 0:
                                met.gx_amp *= -1
                            met.gx_pre.amplitude *= -1
                        if self.alt_pha is True:
                            met.gy_pre.amplitude *= -1
                            met.gy_blip_amp *= -1
                        if self.alt_slc is True and self.acq_3d is True and met.size[2] > 1:
                            met.gz_pre_amp *= -1

                    #Loop through second phase encodes/slices
                    tv_delay = self.tv
                    for z_idx, z in enumerate(met.z_range):

                        #Make slice event with spatial offset if necessary
                        block_start = len(self.block_durations)

                        if met.use_sinc is False:

                            if self.acq_3d is False:

                                #Update ssrf for current slice
                                pha_off = 2 * np.pi * met.kz * \
                                          (self.fov[2] / met.size[2] * z + self.slc_off)
                                ssrf = met.mag * np.exp(1j * (met.pha + pha_off))
                                met.rf.signal = ssrf

                            #Add ssrf block
                            try:
                                self.add_blck_lbl(met.rf_gz, met.rf, met.rf_delay, lbl=m)
                            except AttributeError:
                                self.add_blck_lbl(met.rf_gz, met.rf, lbl=m)

                        else:

                            #Adjust slice frequency offset
                            if self.acq_3d is False:
                                met.rf.freq_offset = met.rf_freq_offs[z_idx]

                            #Add rf block. Tried to wait until prephasers to do rephasing
                            #But had issue with z encoding that way
                            self.add_blck_lbl(met.rf, met.rf_gz, lbl=m)
                            self.add_blck_lbl(met.rf_gz_r, lbl=m)

                        #Echo loop
                        for echo in range(self.n_echo):

                            #Prephasing block setup
                            if self.no_pe is True:
                                met.gy_pre.amplitude = 0
                                if self.symm_ro is True:
                                    met.gy_up.waveform *= 0
                                    met.gy_dn.waveform *= 0
                                    met.gy.waveform *= 0
                                else:
                                    met.gy.amplitude = 0
                                if self.acq_3d is True and met.size[2] > 1:
                                    met.gz_pre.amplitude = 0
                            else:

                                #Flip phase encodes for multi echo
                                if self.n_echo > 1:
                                    if echo % 2 == 0:
                                        sign = 1
                                    else:
                                        sign = -1
                                    if self.symm_ro is True:
                                        gy_peak_idx = np.argmax(np.abs(met.gy.waveform))
                                        gy_scale = met.gy.waveform[gy_peak_idx] * sign
                                        met.gy.waveform *= met.gy_blip_amp / gy_scale
                                        met.gy_up.waveform *= met.gy_blip_amp / gy_scale
                                        met.gy_dn.waveform *= met.gy_blip_amp / gy_scale
                                    else:
                                        met.gy.amplitude = met.gy_blip_amp * sign

                                #Step up/down second phase encoding dimension
                                if self.acq_3d is True and met.size[2] > 1:
                                    pre_amp = met.gz_pre_amp * z      #* (1 - z * 2 / (n_z[m]))
                                    met.gz_pre.amplitude = pre_amp * (not self.no_slc)

                            #Add prephasing block if necessary
                            if echo == 0:
                                if self.acq_3d is False or met.size[2] == 1:
                                    if met.use_sinc is False:
                                        self.add_blck_lbl(met.gx_pre, met.gy_pre, lbl=m)
                                    else:
                                        self.add_blck_lbl(met.gx_pre, met.gy_pre, lbl=m)
                                else:
                                    if met.use_sinc is False:
                                        #self.add_blck_lbl(pp.make_delay(10E-6), lbl=m)
                                        self.add_blck_lbl(met.gx_pre, met.gy_pre, met.gz_pre, lbl=m)
                                    else:
                                        self.add_blck_lbl(met.gx_pre, met.gy_pre, met.gz_pre, lbl=m)

                            #Loop through phase encodes
                            ro_start = len(self.block_durations)
                            for i in range(met.size_acq[1]):

                                #Flip readout polarity if necessary
                                if self.symm_ro is True and i % 2 == 1:
                                    met.gx.amplitude = met.gx_amp * -1
                                else:
                                    met.gx.amplitude = met.gx_amp

                                #Add data acquisition and readout gradient
                                if self.symm_ro is True and met.size_acq[1] > 1:
                                    if i == met.size_acq[1] - 1:
                                        self.add_blck_lbl(met.adc, met.gx, met.gy_dn, lbl=m)
                                    elif i == 0:
                                        self.add_blck_lbl(met.adc, met.gx, met.gy_up, lbl=m)
                                    else:
                                        self.add_blck_lbl(met.adc, met.gx, met.gy, lbl=m)
                                else:
                                    self.add_blck_lbl(met.adc, met.gx, lbl=m)

                                #Add flyback gradient and phase blip if necessary
                                cond_1 = i != met.size_acq[1] - 1
                                cond_2 = i == met.size_acq[1] - 1 and echo != self.n_echo - 1
                                if cond_1 and self.symm_ro is False:
                                    self.add_blck_lbl(met.gy, met.fly, lbl=m)
                                elif cond_2 and self.symm_ro is False:
                                    self.add_blck_lbl(met.fly, lbl=m)

                            #Add a delta te delay if necessary
                            if self.n_echo > 1:
                                ro_dur = np.sum(self.block_durations[ro_start::])
                                delta_delay = self.delta_te - ro_dur
                                if delta_delay > 0:
                                    delta_delay = np.ceil(delta_delay / \
                                                          self.system.block_duration_raster) * \
                                                  self.system.block_duration_raster
                                    self.add_blck_lbl(pp.make_delay(delta_delay), lbl=m)

                        #Spoilers if requested
                        if self.grad_spoil is True:
                            met.x_spoil.amplitude = met.spoil_amp * np.sign(met.gx.amplitude)
                            if self.symm_ro is True:
                                met.y_spoil.amplitude = met.spoil_amp
                            else:
                                met.y_spoil.amplitude = met.spoil_amp * np.sign(met.gy.amplitude)
                            if self.acq_3d is True and met.size[2] > 1:
                                met.z_spoil.amplitude = -met.gz_pre.amplitude
                            self.add_blck_lbl(met.x_spoil, met.y_spoil, met.z_spoil, lbl=m)

                        #Add TR delay
                        exc_dur = np.sum(self.block_durations[block_start::])
                        tv_delay -= exc_dur
                        ts_delay -= exc_dur
                        if self.tr > 0:
                            tr_delay = self.tr - exc_dur
                            tr_delay = np.ceil(tr_delay / \
                                               self.system.block_duration_raster) * \
                                       self.system.block_duration_raster
                            if tr_delay > 0:
                                self.add_blck_lbl(pp.make_delay(tr_delay), lbl=m)
                                tv_delay -= tr_delay
                                ts_delay -= tr_delay

                    #Add volume delay
                    if tv_delay > 0:
                        self.add_blck_lbl(pp.make_delay(tv_delay), lbl=m)
                        ts_delay -= tv_delay

                #Add metabolite set delay
                if ts_delay > 0:
                    self.add_blck_lbl(pp.make_delay(ts_delay), lbl=m)

                if a == 0 and r == 0 and return_plot is True:
                    plot_seq = deepcopy(self)

            #Add spectra if necessary
            try:
                if self.spec.run in ('END', 'BOTH'):
                    for i in range(self.spec.n):
                        if i == 0:
                            spec_start = len(self.block_durations)
                        self.add_blck_lbl(self.spec.rf, self.spec.gz, lbl='s')
                        self.add_blck_lbl(self.spec.gz_r, lbl='s')
                        self.add_blck_lbl(self.spec.adc, lbl='s')
                        if i == 0:
                            spec_dur = np.sum(self.block_durations[spec_start::])
                            spec_delay = self.spec.tr - spec_dur
                        if spec_delay > 0:
                            self.add_blck_lbl(pp.make_delay(spec_delay), lbl='s')
            except AttributeError:
                pass

        #Return a copy of the first few replications
        if return_plot is True:
            return plot_seq
        return None

    def create_param_dic(self):
        """
        Create a dictionary for sequence parameters

        Returns
        -------
        param_dic : dic
           Dictionary of sequence parameters
        """

        #Generic parameters
        out_dic = {"fov":[int(dim * 1E3) for dim in self.fov],
                   "rbw":self.rbw,
                   "n_avg":self.n_avg,
                   "n_rep":self.n_rep,
                   "tr":self.tr * 1E3,
                   "tv":self.tv * 1E3,
                   "ts":self.ts * 1E3,
                   "ro_off":self.ro_off * 1E3,
                   "pe_off":self.pe_off * 1E3,
                   "slc_off":self.slc_off * 1E3,
                   "n_echo":self.n_echo,
                   "delta_te":self.delta_te * 1E3,
                   "n_met":self.n_met,
                   "symm_ro":self.symm_ro,
                   "acq_3d":self.acq_3d,
                   "no_pe":self.no_pe,
                   "no_slc":self.no_slc,
                   "grad_spoil":self.grad_spoil,
                   "ramp_samp":self.ramp_samp,
                   "slice_axis":self.slice_axis.upper(),
                   "alt_read":self.alt_read,
                   "alt_pha":self.alt_pha,
                   "alt_slc":self.alt_slc,
                   "ro_os":self.ro_os,
                   "b0":self.system.B0,
                   "nuc":self.nuc,
                   "max_grad":self.system.max_grad / self.system.gamma * 1E3,
                   "max_slew":self.system.max_slew / self.system.gamma,
                   "rf_ringdown_time":self.system.rf_ringdown_time * 1E6,
                   "rf_dead_time":self.system.rf_dead_time * 1E6,
                   "ori":self.ori,
                   "pe_dir":self.pe_dir}

        #Spectra parameters
        try:
            out_dic['run_spec'] = self.spec.run
            out_dic['spec_size'] = self.spec.size
            out_dic['spec_bw'] = self.spec.bw / 1E3
            out_dic['spec_flip'] = self.spec.flip / np.pi * 180
            out_dic['spec_tr'] = self.spec.tr * 1E3
            out_dic['spec_n'] = self.spec.n
        except AttributeError:
            pass

        #Update metabolites whose parameters were actually set
        out_dic['mets'] = []
        for met in self.mets:
            met_dic = {}
            met_dic['name'] = met.name
            met_dic['grd_path'] = met.grd_path
            met_dic['rf_path'] = met.rf_path
            met_dic['formula'] = met.formula
            met_dic['use_sinc'] = met.use_sinc
            met_dic['flip'] = met.flip / np.pi * 180
            met_dic['freq_off'] = met.freq_off
            met_dic['sinc_dur'] = met.sinc_dur * 1E3
            met_dic['sinc_frac'] = met.sinc_frac
            met_dic['sinc_tbw'] = met.sinc_tbw
            met_dic['size'] = met.size
            met_dic['pf_pe'] = met.pf_pe
            met_dic['pf_pe2'] = met.pf_pe2
            met_dic['esp'] = met.esp
            met_dic['b1_max'] = met.b1_max
            met_dic['rf_delta'] = met.rf_delta * 1E6
            met_dic['grd_max'] = met.grd_max
            met_dic['grd_delta'] = met.grd_delta * 1E6
            met_dic['z_centric'] = met.z_centric
            met_dic['size_acq'] = [int(dim) for dim in met.size_acq]
            out_dic['mets'].append(met_dic)

        return out_dic

    def save_params(self, out_path):
        """
        Saves input parameters that were used to define section to json

        Parameters
        ----------
        out_path : str
           Path to write json data to

        Returns
        -------
        out_dic : json
           Json file containing parameters
        """

        #Get parameter dictionary
        if hasattr(self, 'param_dic') is True:
            out_dic = self.param_dic
        else:
            out_dic = self.create_param_dic()

        #Save json data
        with open(f'{out_path}.json', "w", encoding='utf-8') as fid:
            json.dump(out_dic, fid, indent=2)
