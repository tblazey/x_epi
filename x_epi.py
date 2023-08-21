#!/usr/bin/python

#Load libraries
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pypulseq as pp
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
import re
import scipy.integrate as integ
import scipy.interpolate as interp
import sys
import types

max_met = 5

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
   
   match nuc: 
      case "13C":
         gamma = 10.7084E6
      case "1H":
         gamma = 42.57638474E6
      case "2H":
         gamma = 6.536E6
      case "15N":
         gamma = -4.316E6
      case "17O":
         gamma = -5.722E6
      case "31P":
         gamma = 17.235E6
      case "19F":
         gamma = 40.078E6
      case "23Na":
         gamma = 11.262E6
      case "129Xe":
         gamma = -11.777E6
      case _:
         raise ValueError('Unknown nucleus')
   
   return gamma

def load_ssrf_grd(grd_path):
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

   #Open file to get maximum gradient strength and time resolution
   with open(grd_path) as fid:
      grd_txt = fid.read()
   grd_max = float(re.search('Max Gradient Strength = .*', grd_txt)[0].split()[4])
   grd_delta = float(re.search('Resolution = .*', grd_txt)[0].split()[2]) * 1E-6
   
   #Load in gradient data. A bit inefficient, but more readable
   grd_data = np.loadtxt(grd_path, usecols=[0], comments='#')

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

   #Open file to get maximum b1 and time resolution
   with open(rf_path) as fid:
      rf_txt = fid.read()
   b1_max = float(re.search('Max B1 = .*', rf_txt)[0].split()[3])
   rf_delta = float(re.search('Resolution = .*', rf_txt)[0].split()[2]) * 1E-6
   
   #Load in rf data. A bit inefficient, but more readable
   rf_data = np.loadtxt(rf_path, usecols=[0, 1], comments='#')
   
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
   ti_stop : float
      Time to end interpolation at
   
   Returns
   -------
   sig_i : ndarray
      Array containing interpolated data
   """
   
   #Get time vectors
   n = sig.shape[0]
   t = np.arange(1, n + 1) * delta_t
   if ti_end is None:
      ti_end = np.ceil(t[-1] / delta_ti) * delta_ti
   ti = np.arange(delta_ti, ti_end + delta_ti, delta_ti)
      
   #Interpolate
   return interp.interp1d(t, sig, bounds_error=False, fill_value=0)(ti)

def compute_k_space(seq, n_met):
      """
      Computes rf, gradient, and k-space waveforms of current sequence
      
      Parameters
      ----------
      seq : xEPI object
         Sequence object containing waveforms
      n_met : int
         Number of metabolites
         
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
   
      #Get waveforms
      [wave_data, tfp_exc, tfp_ref, t_adc, fp_adc] = seq.waveforms_and_times(append_RF=True)

      #Resample gradient times so that they are all the same
      t_max = [wave_data[0][0, -1], wave_data[1][0, -1]]
      if wave_data[2].shape[1] > 0:
            t_max.append(wave_data[2][0, -1])
      t_max = np.max(t_max)
      t_i = np.arange(0, t_max + seq.system.grad_raster_time, seq.system.grad_raster_time)
      x_i = interp.interp1d(wave_data[0][0, :], wave_data[0][1, :], fill_value=0.0, 
                            bounds_error=False)(t_i)
      y_i = interp.interp1d(wave_data[1][0, :], wave_data[1][1, :], fill_value=0.0, 
                            bounds_error=False)(t_i)
      if wave_data[2].shape[1] > 0:
         z_i = interp.interp1d(wave_data[2][0, :], wave_data[2][1, :], fill_value=0.0, 
                               bounds_error=False)(t_i)
      else:
         z_i = np.zeros_like(t_i)
      rf_i = interp.interp1d(np.real(wave_data[3][0, :]), wave_data[3][1, :],
                             fill_value=0.0, bounds_error=False)(t_i)                       

      #Find right edges of rf/blocks
      rf_edges = [[] for i in range(n_met)]
      rf_durs = [[] for i in range(n_met)]
      block_edges = np.cumsum(seq.block_durations)                    
      for i in range(1, len(seq.block_events) + 1):
         rf_num = seq.block_events[i][1]
         met_idx = seq.blck_lbls[i - 1]
         if rf_num > 0 and type(met_idx) is int:
            rf_edges[met_idx].append(block_edges[i - 1])
            rf_durs[met_idx].append(seq.block_durations[i - 1])
  
      #Make empty arrays for storing data
      k_list = []                #k-space points in 3d plane
      k_2d_list = []             #k-space points in 2d plane
      adc_list = []              #adc times in 3d plane
      adc_2d_list = []           #adc times in 2d plane
      g_list = []                #interpolated gradients
      rf_list = []               #interpolated rf waveforms
         
      #Loop through metabolites
      for m in range(n_met):
                  
         #Compute k-space, restart integration at each rf edge
         k_x = np.zeros(t_i.shape[0])
         k_y = np.zeros_like(k_x)
         k_z = np.zeros_like(k_x)
         k_mask = np.zeros(k_x.shape[0], dtype=bool)
         for idx, edge in enumerate(rf_edges[m]):
            if idx == len(rf_edges[m]) - 1:
               if m == n_met - 1:
                  mask = t_i > rf_edges[m][idx]
                  if idx == 0:
                     grad_mask =  t_i > (rf_edges[m][idx] - rf_durs[m][0])
               else:
                  mask = np.logical_and(t_i > rf_edges[m][idx], 
                                        t_i <= (rf_edges[m + 1][0] - rf_durs[m + 1][0]))
                  if idx == 0:
                     grad_mask = np.logical_and(t_i > (rf_edges[m][idx]- rf_durs[m][0]), 
                                                t_i <= (rf_edges[m + 1][0] - rf_durs[m + 1][0]))
            else:
               start_mask = t_i >= (rf_edges[m][idx] - rf_durs[m][idx])
               end_mask = t_i <= (rf_edges[m][idx + 1] - rf_durs[m][idx + 1])
               if idx == 0:
                  grad_mask = np.logical_and(start_mask, end_mask)
               mask = np.logical_and(t_i > rf_edges[m][idx], end_mask) 
            k_x[mask] = integ.cumulative_trapezoid(x_i[mask], t_i[mask], initial=0.0)
            k_y[mask] = integ.cumulative_trapezoid(y_i[mask], t_i[mask], initial=0.0)
            k_z[mask] = integ.cumulative_trapezoid(z_i[mask], t_i[mask], initial=0.0)
            if idx == 0:
               k_mask_2d = np.copy(mask)
            k_mask = np.logical_or(k_mask, mask)
   
         #Figure out k-space locations for each adc sample time
         t_mask = np.logical_and(t_adc >= np.min(t_i[k_mask]), t_adc <= np.max(t_i[k_mask]))
         t_mask_2d = np.logical_and(t_adc[t_mask] >= np.min(t_i[k_mask_2d]),
                                    t_adc[t_mask] <= np.max(t_i[k_mask_2d]))
         adc_x = interp.interp1d(t_i[k_mask],  k_x[k_mask])(t_adc[t_mask])
         adc_y = interp.interp1d(t_i[k_mask],  k_y[k_mask])(t_adc[t_mask])
         adc_z = interp.interp1d(t_i[k_mask],  k_z[k_mask])(t_adc[t_mask])
         k_list.append(np.stack((k_x[k_mask], k_y[k_mask], k_z[k_mask])))
         k_2d_list.append(np.stack((k_x[k_mask_2d], k_y[k_mask_2d])))
         adc_list.append(np.stack((adc_x, adc_y, adc_z)))
         adc_2d_list.append(np.stack((adc_x[t_mask_2d], adc_y[t_mask_2d])))
         g_list.append(np.stack((x_i[grad_mask], y_i[grad_mask], z_i[grad_mask])))
         rf_list.append(rf_i[grad_mask])
         
      #Return waveforms
      return k_list, adc_list, k_2d_list, adc_2d_list, g_list, rf_list

class xEPI(pp.Sequence):
   def __init__(self, fov=[240, 240, 240], rbw=50, n_avg=1, n_rep=1, tr=0, 
                tv=0, ts=0, ro_off=0, pe_off=0, slc_off=0, slice_axis='Z',
                n_echo=1, delta_te=0, symm_ro=True, acq_3d=True, no_pe=False,
                grad_spoil=False, no_slc=True, alt_read=False, alt_pha=False,
                alt_slc=False, b0=3, nuc='13C', rf_dead_time=100, rf_ringdown_time=30,
                max_slew=200, max_grad=100, ori='Transverse', pe_dir='AP', **kwargs):
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
      X EPI sequence object
      """
      
      #Initialize sequence object
      lims = pp.Opts(max_grad=max_grad, grad_unit="mT/m", max_slew=max_slew,
                     slew_unit="T/m/s", rf_ringdown_time=rf_ringdown_time * 1E-6,
                     rf_dead_time=rf_dead_time * 1E-6, gamma=nuc_to_gamma(nuc),
                     B0=b0, adc_dead_time=20E-6)
      super().__init__(lims)

      #Primary x_epi specific sequence options
      self.nuc = nuc
      self.fov = np.array(fov) / 1E3            #square FOV (meters)
      self.rbw = rbw
      self.n_avg = n_avg                        #number of averages
      self.n_rep = n_rep                        #number of repetitions
      self.tr = tr / 1E3                        #minimum duration of each excitation block (s)
      self.tv = tv / 1E3                        #minimum duration of each 3D image (s)
      self.ts = ts / 1E3                        #minimum duration of each metabolite set (s)
      self.ro_off = ro_off / 1E3                #spatial shift in readout direction (m)
      self.pe_off = pe_off / 1E3                #spatial shift in phase encoding direction
      self.slc_off = slc_off / 1E3              #spatial shift in slice direction (m)  
      self.slice_axis = slice_axis.lower()      #axis to play slice gradients on
      self.n_echo = n_echo                      #number of echos following each excitation
      self.delta_te = delta_te / 1E3            #minimum timing between echos (s)
      self.n_met = 0                            #initialize number of metabolites
      self.mets = []                            #list of metabolite objects
      self.symm_ro = symm_ro                    #True=symmetric, False=flyback
      self.acq_3d = acq_3d                      #True = 3D, False = 2D
      self.no_pe = no_pe                        #True = No phase encoding grads
      self.grad_spoil = grad_spoil              #True = Gradient spoilers
      self.no_slc = no_slc                      #True = No slice grads
      self.alt_read = alt_read                  #True = Alternate polarity of readout every repetition
      self.alt_pha = alt_pha                    #True = Alternate polarity of phase encoding every repetition
      self.alt_slc = alt_slc                    #True = Alternate polarity of second phase encoding every repetition      
      self.ori = ori                            #Image orientation
      self.pe_dir = pe_dir                      #Phase encoding direction
      
      #Compute parameters that don't vary between metabolites
      self.delta_v = self.rbw / 2 * 1E3               #half the receiver bandwidth (hz)
      self.dwell = 1 / self.delta_v                   #Time between readout points (s)
      self.gx_amp = 2 * self.delta_v / self.fov[0]    #Readout amplitude (Hz/m) 
      
      #Create empty list for labeling blocks
      self.blck_lbls = []        
      
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
      self.spec.run = run_spec                  #when to run spectra
      self.spec.size = spec_size                #number of complex points to acquire
      self.spec.bw = spec_bw * 1E3              #spectral bandwidth (Hz)
      self.spec.flip = spec_flip / 180 * np.pi  #radians
      self.spec.tr = spec_tr / 1E3              #minimum TR for each spectra acquisition (s)
      self.spec.n = n_spec                      #number of spectra to acquire
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
      self.spec.rf.freq_offset += self.spec.gz.amplitude * self.slc_off
         
      #Make adc event                                                                                                         
      self.spec.adc = pp.make_adc(num_samples=self.spec.size, system=self.system,
                                  delay=self.system.adc_dead_time,
                                  duration=self.spec.size / self.spec.bw)

   def add_met(self, name=None, size=[16, 16, 16], pf_pe=1, pf_pe2=1,
               sinc_frac=0.5, sinc_tbw=4, grd_path='siemens_pyr_plateau_slab.GRD',
               rf_path='siemens_pyr_plateau_slab.RF', formula='1', use_sinc=False,
               flip=90, freq_off=0, sinc_dur=4, **kwargs):
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
      """
      
      #Figure out names
      self.n_met += 1
      if name is None:
         name = f'met_{self.n_met}'
      
      #Add metabolite specific options
      met_obj = types.SimpleNamespace()
      met_obj.name = name                       #Metabolite name
      met_obj.size = size                       #Size of metabolite grid
      met_obj.pf_pe = pf_pe                       #Partial Fourier fraction in y dimension
      met_obj.pf_pe2 = pf_pe2                       #Partial Fourier fraction in z dimension
      met_obj.use_sinc = use_sinc               #Are we using a sinc pulse or an ssrf?
      met_obj.flip = flip / 180 * np.pi         #Flip angle in radians
      met_obj.freq_off = freq_off               #Pulse frequency offset in Hz
      met_obj.sinc_frac = sinc_frac             #Center fraction of sinc pulse
      met_obj.sinc_tbw = sinc_tbw               #Time-bandwidth fraction of sinc pulse
      met_obj.sinc_dur = sinc_dur * 1E-3        #Duration of sinc pulse (s)
      met_obj.grd_path = grd_path               #Path to SSRF gradient shape
      met_obj.rf_path = rf_path                 #Path to SSRF RF shape
      met_obj.formula = formula                 #SSRF gradient scale formula
      met_obj.b1_max = 0
      met_obj.rf_delta = 0
      met_obj.grd_max = 0
      met_obj.grd_delta = 0
      
      #Grid grid size that will be acquired
      met_obj.size_acq = np.int32(np.round(np.array(met_obj.size) * \
                                           np.array([1, met_obj.pf_pe, met_obj.pf_pe2])))
                                  
      #Construct readout gradient (in Hz/m)
      t_acq = met_obj.size[0] / self.delta_v / 2;
      flat_time = np.ceil(t_acq / self.system.grad_raster_time) * \
                  self.system.grad_raster_time
      met_obj.gx = pp.make_trapezoid(channel='x', system=self.system, flat_time=flat_time,
                                     amplitude=self.gx_amp)
      
      #Excitation thickness
      if self.acq_3d is True:
         met_obj.exc_thk = self.fov[2]
      else:
         met_obj.exc_thk = self.fov[2] / met_obj.size[2]
      
      #Make flyback readout if needed
      if self.symm_ro is False:
         met_obj.fly = pp.make_trapezoid('x', system=self.system, area=-met_obj.gx.area)
         
      """
      Data acquisition event. Delay is constructed so that it includes ramp time as well
      as adjustments for the rounding that had to occur to match gradient raster and the 
      fact that ADC samples take place in the middle of trapezoidal gradient times. Extra 
      time for rounding is divided evenly on each side of the readout gradient. The dwell
      time can be removed from remove (t_acq - t_dwell) if you want no odd/even offset
      """
      adc_delay = met_obj.gx.rise_time + flat_time / 2 - (t_acq) / 2
      met_obj.adc = pp.make_adc(met_obj.size[0], duration=t_acq, delay=adc_delay,
                                system=self.system,
                                freq_offset=met_obj.freq_off + self.ro_off * self.gx_amp)
      
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
         
      #If using symmetric readout, start the phase blip during readout descending ramp
      if self.symm_ro is False:
         gy_delay = 0;
      else:
         gy_delay = met_obj.gx.rise_time + met_obj.gx.flat_time;

      #Construct basic phase encoding blip
      gy_blip = pp.make_trapezoid('y', system=self.system, area= 1 / self.fov[1],
                                  delay=gy_delay)
      met_obj.gy_blip_amp = gy_blip.amplitude
      gy_dur = gy_blip.rise_time + gy_blip.fall_time + gy_blip.flat_time
      if gy_dur > (met_obj.gx.rise_time + met_obj.gx.fall_time) and self.symm_ro is True:
         raise Exception('Phase encoding blip cannot fit in between readout plateaus') 

      #Split phase encoding gradient if necessary
      if self.symm_ro is True:
         gy_blip.delay = 0
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
         met_obj.z_range = np.linspace(1, -1, met_obj.size[2])[0:int(met_obj.size_acq[2])][::-1]       
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
         met_obj.rf.freq_offset = met_obj.freq_off + met_obj.rf_gz.amplitude * self.slc_off
         
         #Create frequency slices for slices
         if self.acq_3d is False:
            
            #Loop through slice multipliers
            met_obj.rf_freq_offs = []
            for z in range(met_obj.z_range):
            
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
   
      #Convert magnitude to Hz and phase to radians. Factor 2 / np.pi is to make it a fraction of 90 degree pulse
      mag *= met_obj.b1_max * 2 * met_obj.flip / np.pi / np.max(mag) * 1E-4 * self.system.gamma
      pha *= np.pi / 180
   
      #Interpolate rf data to rf raster time
      rf_end = (mag.shape[0] - 1) * met_obj.rf_delta
      rf_end = np.ceil(rf_end / self.system.grad_raster_time) * self.system.grad_raster_time
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
      grd_data, met_obj.grd_max, met_obj.grd_delta = load_ssrf_grd(met_obj.grd_path)
         
      #Convert gradient data to Hz / m
      grd_data *= met_obj.grd_max / np.max(np.abs(grd_data)) * 1E-2 * self.system.gamma
         
      #Scale gradient using slice/slab thickness formula
      met_obj.slc_scale = eval(met_obj.formula, {'x':met_obj.exc_thk * 1E3})
      grd_data *= met_obj.slc_scale
   
         
      #Interpolate gradient waveform to raster
      grd_data_i = interp_waveform(grd_data * (not self.no_slc), met_obj.grd_delta, 
                                   self.system.grad_raster_time)
                                
      #Interpolate k-space vector to rf raster time
      kz = integ.cumtrapz(grd_data, dx=met_obj.grd_delta, initial=0)
      kz_end = (grd_data.shape[0] - 1) * met_obj.grd_delta
      kz_end = np.ceil(kz_end / self.system.grad_raster_time) * self.system.grad_raster_time
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
            if self.spec.run == 'START' or self.spec.run == 'BOTH':
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
         except:
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
                        self.gx_amp *= -1
                     met.gx_pre.amplitude *= -1
                  if self.alt_phase is True:
                     met.gy_pre.amplitude *= -1
                     met.gy_blip_amp *= -1
                  if self.alt_slc is True and self.acq_3d is True and met.size[2] > 1:
                     gz_pre_amp *= -1
            
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
                     except:
                        self.add_blck_lbl(met.rf_gz, met.rf, lbl=m)
                     
                  else:
               
                     #Adjust slice frequency offset
                     if self.acq_3d is False:
                        met.rf.freq_offset = met.rf_freq_offs[z_idx]
                  
                     #Add rf block. Wait to do rephasing gradient
                     self.add_blck_lbl(met.rf, met.rf_gz, lbl=m)
            
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
                              self.add_blck_lbl(met.gx_pre, met.gy_pre,
                                                    met.rf_gz_r, lbl=m)
                        else:
                           if met.use_sinc is False:
                              self.add_blck_lbl(pp.make_delay(10E-6), lbl=m)
                              self.add_blck_lbl(met.gx_pre, met.gy_pre, met.gz_pre, lbl=m)
                           else:
                              self.add_blck_lbl(met.gx_pre, met.gy_pre, met.gz_pre,
                                                    met.rf_gz_r, lbl=m)
         
                     #Loop through phase encodes
                     ro_start = len(self.block_durations)
                     for i in range(met.size_acq[1]):
         
                        #Flip readout polarity if necessary
                        if self.symm_ro is True and i % 2 == 1:
                           met.gx.amplitude = self.gx_amp * -1
                        else:
                           met.gx.amplitude = self.gx_amp

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
                  
                        #Add flack gradient and phase blip if necessary
                        if (i !=  met.size_acq[1] - 1 or (i == met.size_acq[1] - 1 and echo != self.n_echo - 1)) and self.symm_ro is False:
                           self.add_blck_lbl(met.gy, met.fly, lbl=m)
                        
                     #Add a delta te delay if necessary
                     if self.n_echo > 1:
                        ro_dur = np.sum(self.block_durations[ro_start::])
                        delta_delay = self.delta_te - ro_dur
                        if delta_delay > 0:
                           delta_delay = np.ceil(delta_delay / self.system.block_duration_raster) * self.system.block_duration_raster
                           self.add_blck_lbl(pp.make_delay(delta_delay), lbl=m)
                                
                  #Spoilers if requested
                  if self.grad_spoil is True:
                     met.x_spoil.amplitude = met.spoil_amp * np.sign(met.gx.amplitude)
                     if self.symm_ro is True:
                         met.y_spoil.amplitude = met.spoil_amp * -1
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
                     tr_delay = np.ceil(tr_delay / self.system.block_duration_raster) * self.system.block_duration_raster
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
            if self.spec.run == 'END' or self.spec.run == 'BOTH':
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
         except:
            pass
            
      #Return a copy of the first few replications
      if return_plot is True:
         return plot_seq
   
   def create_param_dic(self):
      """
      Create a dictionary for sequence parameters
      
      Returns
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
                 "slice_axis":self.slice_axis.upper(),
                 "alt_read":self.alt_read,
                 "alt_pha":self.alt_pha,
                 "alt_slc":self.alt_slc,
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
      except:
         pass
      
      #Update metabolites whose parameters were actually set
      out_dic['mets'] = []
      for idx, met in enumerate(self.mets):
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
      with open('%s.json'%(out_path), "w") as fid:
         json.dump(out_dic, fid, indent=2)
