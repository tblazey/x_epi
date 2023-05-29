#!/usr/bin/python

#Load libraries
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
import pypulseq as pp
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
import re
import scipy.integrate as integ
import scipy.interpolate as interp

def create_seq(win, return_plot=True, only_plot=False):

   #Set system limits and create sequence object
   lims = pp.Opts(max_grad=win.dbl_spin_grd_limit.value(),
                  grad_unit="mT/m",
                  max_slew=win.dbl_spin_slew_limit.value(),
                  slew_unit="T/m/s",
                  rf_ringdown_time=win.dbl_spin_ringdown.value() * 1E-6,
                  rf_dead_time=win.dbl_spin_dead_time.value() * 1E-6,
                  gamma=win.gamma,
                  B0=win.dbl_spin_b0.value(),
                  adc_dead_time=20E-6)
   seq = pp.Sequence(system=lims)

   #Extract sequence parameters
   fov = np.array([win.spin_fov_x.value(), 
                   win.spin_fov_y.value(),
                   win.spin_fov_z.value()]) / 1E3     #square FOV (meters)
   n_x = win.size[:, 0]                               #number of readout points                
   n_y = win.size[:, 1]                               #number of phase encodes
   n_z = win.size[:, 2]                               #number of phase encodes in second dimension/slices
   delta_v = win.spin_band.value() / 2 * 1E3          #half the receiver bandwidth (Hz)
   n_avg = win.spin_avg.value()                       #number of averages
   n_rep = win.spin_rep.value()                       #number of repetitions
   tr = win.dbl_spin_tr.value() / 1E3                 #total duration of each excitation block. 0 = minimum (s)
   tv = win.dbl_spin_tv.value() / 1E3                 #total duration of each 3D image. 0 = minimum (s)
   ts = win.dbl_spin_ts.value() / 1E3                 #total duration of each metabolite set. 0 = minimum (s)
   ky_frac = win.pf[:, 0]                             #partial Fourier fraction in 1st phase encoding dimension
   kz_frac = win.pf[:, 1]                             #partial Fourier fraction in 2nd phase encoding dimension
   freq_offset = win.freq_offset                      #Frequency offset for RF pulses and ADC in Hz                          
   rf_paths = win.rf_paths
   grd_paths = win.grd_paths
   flips = win.flips
   n_met = win.spin_met.value()
   rf_delta = win.rf_delta
   rf_types = win.rf_types
   grd_delta = win.grd_delta
   max_b1 = win.max_b1
   max_grd = win.max_grd
   slice_axis = win.combo_slice_axis_sel.currentText().lower()
   grd_forms = win.grd_forms
   
   #Extract sequence options
   if win.button_group_readout.checkedButton().text() == "Symmetric":
      readout = 0
   else:
      readout = 1
   if win.button_group_mode.checkedButton().text() == "2D":
      acq_3d = 0
   else:
      acq_3d = 1
   if win.button_group_phase.checkedButton().text() == "Off":
      no_pe = 1
   else:
      no_pe = 0
   if win.button_group_spoil.checkedButton().text() == "Off":
      grad_spoil = 0
   else:
      grad_spoil = 1 
   if win.button_group_skip.checkedButton().text() == "On" and readout == 0:
      skip_acq = 1
   else:
      skip_acq = 0
   if win.button_group_slice.checkedButton().text() == "Off":
      slice_on = 0
   else:
      slice_on = 1
   if win.button_group_scale.checkedButton().text() == "Off":
      no_scale = 1
   else:
      no_scale = 0
      
   #Define sequence labels
   srt_lbl = pp.make_label(type='SET', label='ONCE', value=1)
   end_lbl = pp.make_label(type='SET', label='ONCE', value=2)
   slc_lbl = pp.make_label(type='SET', label="NOSLC", value=no_scale)
  
   #If we are just doing this for making a plot, turn off repetitions and averages
   if only_plot is True:
      n_rep = 1
      n_avg = 1

   #Compute total numbers of steps that we will actually acquire
   n_y_acq = np.round(ky_frac * n_y)
   n_z_acq = np.round(kz_frac * n_z)

   #Compute values that are the same for each metabolite
   t_dwell = 1.0 / 2.0 / delta_v;
   gx_amp = 2 * delta_v / fov[0];

   #Make empty lists for storing sequence items
   ssrf_grd_ev = []
   ssrf_kz = []
   ssrf_mag = []
   ssrf_phase = []
   gx_ev = []
   gy_ev = []
   gy_up_ev = []
   gy_dn_ev = []
   gx_pre_ev = []
   gy_pre_ev = []
   if readout == 1:
      fly_ev = []
   if acq_3d == 1:
      gz_pre_ev = []
      gz_pre_amp = np.zeros(n_met)
   if grad_spoil == 1:
      x_spoil_ev = []
      y_spoil_ev = []
      spoil_amp = np.zeros(n_met)
      z_spoil_ev = []
   adc_ev = []
   esp = np.zeros(n_met)
   seq.met_block_id = []
   
   #Are we doing a spectra of any kind?
   if win.check_spectra_start.isChecked() or win.check_spectra_end.isChecked():
   
      #Extract params
      n_points = win.spin_spectra_points.value()
      spec_dwell = 1.0 / win.spin_spectra_band.value() / 1E3
      spec_flip = win.dbl_spin_spectra_flip.value() / 180 * np.pi
      
      #Make rf and adc events
      spec_pulse, spec_gz, spec_gz_r = pp.make_sinc_pulse(flip_angle=spec_flip,
                                                          return_gz=True,
                                                          system=lims,
                                                          slice_thickness=fov[2],
                                                          use='excitation')                                                        
      spec_adc = pp.make_adc(num_samples=n_points, duration=spec_dwell * n_points,
                             delay=lims.adc_dead_time, system=lims)

   #Loop through metabolites
   for i in range(n_met):

      #Load RF pulse if necessary
      if rf_types[i] == 0:
      
         #Load in gradient strength data
         grd_data = np.loadtxt(grd_paths[i], usecols=[0], comments='#')
   
         #Convert slice gradient to Hz / m
         grd_data *= max_grd[i] / np.max(np.abs(grd_data)) * 1E2 / 1E4 * lims.gamma
         
         #Use formula to update gradient scaling for desired slice thickness
         if acq_3d == 1:
            slice_thk = fov[2]
         else:
            slice_thk = fov[2] / n_z[i]
         slice_scale = eval(grd_forms[i],  {'x':slice_thk * 1E3})
         grd_data *= slice_scale
         
         #Compute time vectors for gradient
         n_grd = grd_data.shape[0]
         grd_t = np.arange(1, n_grd + 1) * grd_delta[i]
         grd_ti_end = np.ceil(grd_t[-1] / lims.grad_raster_time) * lims.grad_raster_time
         grd_ti = np.arange(lims.grad_raster_time, grd_ti_end + lims.grad_raster_time,
                            lims.grad_raster_time)
         
         #Compute kz trajectory for ssrf gradient for slice shifting
         kz = integ.cumtrapz(grd_data, grd_t, initial=0)
      
         #Interpolate gradient to system raster time
         grd_data_i = interp.interp1d(grd_t, grd_data * slice_on, bounds_error=False,
                                      fill_value=0.0)(grd_ti)

         #Make gradient event
         grd_ev = make_arbitrary_grad(slice_axis, grd_data_i, system=lims, 
                                      delay=lims.rf_dead_time)
         grd_ev.first = 0
         grd_ev.last = 0

         #Load in rf data
         rf_data = np.loadtxt(rf_paths[i], usecols=[0, 1], comments='#')

         #Convert mag to Hz and phase to radians
         mag_scale = max_b1[i] * flips[i] / 90 / np.max(rf_data[:, 1]) / 1E4 * lims.gamma
         rf_data[:, 1] *= mag_scale
         rf_data[:, 0] *= np.pi / 180
    
         #Interpolate mag and phase
         n_rf = rf_data.shape[0]
         rf_t = np.arange(1, n_rf + 1) * rf_delta[i]
         rf_ti = np.arange(lims.rf_raster_time, grd_ti_end + lims.rf_raster_time,
                            lims.rf_raster_time)
         mag_i = interp.interp1d(rf_t, rf_data[:, 1], fill_value=0, 
                                 bounds_error=False)(rf_ti)
         phase_i = interp.interp1d(rf_t, rf_data[:, 0], fill_value=0, 
                                   bounds_error=False)(rf_ti)
         kz_i = interp.interp1d(grd_t, kz, fill_value=0, bounds_error=False)(rf_ti)

         #Add pulse items to correct lists
         ssrf_grd_ev.append(grd_ev)
         ssrf_kz.append(kz_i)
         ssrf_mag.append(mag_i)
         ssrf_phase.append(phase_i)
      
      else:
      
         #Hack so that ssrf items have the correct length
         ssrf_grd_ev.append(None)
         ssrf_kz.append(None)
         ssrf_mag.append(None)
         ssrf_phase.append(None)
         
      #Construct readout gradient (in Hz/m)
      t_acq = n_x[i] * t_dwell;
      flat_time = np.ceil(t_acq / lims.grad_raster_time) * lims.grad_raster_time
      gx = pp.make_trapezoid(channel='x', system=lims, amplitude=gx_amp, 
                             flat_time=flat_time)
      gx_ev.append(gx)
   
      #Construct flyback gradient
      if readout == 1:
         fly_ev.append(pp.make_trapezoid('x', system=lims, area=-gx.area))
   
      """
      Data acquisition events. Delay is constructed so that it includes ramp time as well
      as adjustments for the rounding that had to occur to match gradient raster and the 
      fact that ADC samples take place in the middle of trapezoidal gradient times. Extra 
      time for rounding is divided evenly on each side of the readout gradient
      """
      adc_delay = gx.rise_time + flat_time / 2 - (t_acq - t_dwell) / 2
      adc = pp.make_adc(n_x[i], duration=t_acq, delay=adc_delay, system=lims,
                        freq_offset=freq_offset[i])            
      adc_ev.append(adc)
   
      #Area of prephaser
      gy_pre_area = -n_y[i] / 2 / fov[1] + 1 / fov[1]
      if ky_frac[i] != 1:
         gy_pre_area += (n_y[i] - n_y_acq[i]) / fov[1]
      
      #Make prephasers
      gx_pre_ev.append(pp.make_trapezoid('x', system=lims, area=-gx.area / 2)) 
      if np.abs(gy_pre_area) > 0:
         gy_pre_ev.append(pp.make_trapezoid('y', system=lims, area=gy_pre_area))
      else:
         gy_pre_grad = pp.make_trapezoid('y', system=lims, area=-gx.area / 2)
         gy_pre_grad.amplitude = 0.0
         gy_pre_ev.append(gy_pre_grad)
      if acq_3d == 1 and n_z[i] > 1:
         gz_pre = pp.make_trapezoid(slice_axis, system=lims, 
                                    area=(n_z[i]) / 2 / fov[2] + 1 / fov[2])
         gz_pre_ev.append(gz_pre)
         gz_pre_amp[i] = gz_pre.amplitude

      #If using symmetric readout, start the phase blip during readout descending ramp
      if readout == 1:
         gy_delay = 0;
      else:
         gy_delay = gx.rise_time + gx.flat_time;
         
      #If using symmetric readout, phase blip must fit between readout plateaus
      gy_blip = pp.make_trapezoid('y', system=lims, area= 1 / fov[1], delay=gy_delay)
      gy_dur = gy_blip.rise_time + gy_blip.fall_time + gy_blip.flat_time
      if gy_dur > (gx.rise_time + gx.fall_time) and readout != 1:
         raise Exception('Phase encoding blip cannot fit in between readout plateaus') 
         
      #Split phase encoding gradient if necessary
      if readout == 0:
         gy_blip.delay = 0
         gy_split = pp.split_gradient_at(gy_blip, gy_dur / 2, system=lims)
         gy_up, gy_dn, _ = pp.align(right=gy_split[0], left=[gy_split[1], gx])
         gy_dnup = pp.add_gradients((gy_dn, gy_up), system=lims)
         gy_up_ev.append(gy_up)
         gy_dn_ev.append(gy_dn)
         gy_ev.append(gy_dnup)
      else:
         gy_ev.append(gy_blip)

      #Create spoilers
      if grad_spoil == 1:
         spoil_area = np.abs(gx.area)
         x_spoil_ev.append(pp.make_trapezoid('x', system=lims, area=spoil_area))
         y_spoil_ev.append(pp.make_trapezoid('y', system=lims, area=spoil_area))
         if acq_3d == 1:
            z_spoil_ev.append(gz_pre_ev[i])
            z_spoil_ev[i].amplitude = -z_spoil_ev[i].amplitude
         else:
            z_spoil_ev.append(pp.make_trapezoid(slice_axis, system=lims, area=-spoil_area))
         spoil_amp[i] = x_spoil_ev[i].amplitude  
      
      #Compute echo spacing
      win.esp[i] = gx.rise_time + gx.fall_time + gx.flat_time
      if readout == 1:
         win.esp[i] += fly_ev[i].rise_time +  fly_ev[i].fall_time  
         
   #Average loops
   for a in range(n_avg):
   
      #Add spectra if necessary
      if win.check_spectra_start.isChecked():
         seq.add_block(spec_pulse, spec_gz, srt_lbl, slc_lbl)
         seq.met_block_id.append('s')
         seq.add_block(spec_gz, srt_lbl, slc_lbl)
         seq.met_block_id.append('s')
         seq.add_block(spec_adc, srt_lbl)
         seq.met_block_id.append('s')
 
      #Loop through repetitions
      for r in range(n_rep):
         ts_delay = ts
   
         #Loop through metabolites
         for m in range(n_met):
         
            #Alternate polarity of first gradient each repetition if necessary
            if r > 0:
               if win.check_read_alt.isChecked():
                  if m == 0:
                     gx_amp *= -1
                  gx_pre_ev[m].amplitude *= -1
               if win.check_phase_alt.isChecked():
                  gy_pre_ev[m].amplitude *= -1
                  gy_ev[m].waveform *= -1
                  if readout == 0:
                     gy_up_ev[m].waveform *= -1
                     gy_dn_ev[m].waveform *= -1
               if win.check_slice_alt.isChecked() and acq_3d == 1 and n_z[m] > 1:
                  gz_pre_amp[m] *= -1
               
            if acq_3d == 1:
      
               #Make rf event for slab selection
               if rf_types[m] == 0:
                  ssrf_slab = ssrf_mag[m] * np.exp(1j * ssrf_phase[m])
                  rf_ev = pp.make_arbitrary_rf(ssrf_slab, 2 * np.pi, system=lims,
                                               norm=False, freq_offset=freq_offset[m])
                                      
                  #Make sure RF event is multiple of block duration
                  rf_dur = rf_ev.dead_time + rf_ev.ringdown_time + rf_ev.shape_dur
                  delay_aug =  np.ceil(rf_dur / lims.block_duration_raster) * \
                                       lims.block_duration_raster
                  if delay_aug > 0:
                     rf_delay = pp.make_delay(delay_aug)
                     
               else:
               
                  rf_ev, rf_gz, rf_gz_r = pp.make_sinc_pulse(flip_angle=flips[m] / 180 * np.pi,
                                                             return_gz=True,
                                                             system=lims,
                                                             slice_thickness=fov[2],
                                                             use='excitation',
                                                             time_bw_product=win.sinc_tbw[m],
                                                             duration=win.sinc_dur[m],
                                                             center_pos=win.sinc_cf[m],
                                                             freq_offset=freq_offset[m]) 
                                                                   
               #Make multipliers for second phase encoding dimension
               z_range = np.arange(n_z_acq[m] - 1, -1, -1)
         
            else:
      
               #Slice ranges
               z_range = np.arange(-np.floor(n_z[m] / 2), np.ceil(n_z[m] / 2))
               
               if rf_types[m] == 1:
                  delta_z = fov[2] / n_z[m]
                  rf_ev, rf_gz, rf_gz_r = pp.make_sinc_pulse(flip_angle=flips[m] / 180 * np.pi,
                                                             return_gz=True,
                                                             system=lims,
                                                             slice_thickness=delta_z,
                                                             use='excitation',
                                                             time_bw_product=win.sinc_tbw[m],
                                                             duration=win.sinc_dur[m],
                                                             center_pos=win.sinc_cf[m],
                                                             freq_offset=freq_offset[m]) 
            
            #Loop through second phase encodes/slices
            tv_delay = tv
            for z in z_range:
         
               #Make slice event with spatial offset if necessary
               block_start = len(seq.block_durations)
               if rf_types[m] == 0:
                  
                  if acq_3d == 0:
                     slice_offset = 2 * np.pi * ssrf_kz[m] * fov[2] / n_z[m] * z
                     ssrf_slice = ssrf_mag[m] * np.exp(1j * (ssrf_phase[m] + slice_offset))
                     rf_ev = pp.make_arbitrary_rf(ssrf_slice, 2 * np.pi, system=lims,
                                                  norm=False,
                                                  freq_offset=freq_offset[m])
                                         
                     #Make sure RF event is multiple of block duration
                     rf_dur = rf_ev.dead_time + rf_ev.ringdown_time + rf_ev.shape_dur
                     delay_aug =  np.ceil(rf_dur / lims.block_duration_raster) * \
                                          lims.block_duration_raster
                     if delay_aug > 0:
                        rf_delay = pp.make_delay(delay_aug)
         
                  #Add ssrf block
                  try:
                     seq.add_block(ssrf_grd_ev[m], rf_ev, rf_delay, slc_lbl)
                     seq.met_block_id.append(m)
                  except:
                     seq.add_block(ssrf_grd_ev[m], rf_ev, slc_lbl)
                     seq.met_block_id.append(m)
               else:
               
                  #Adjust slice frequency offset. Need to check on the 0.5
                  if acq_3d == 0:
                     rf_ev.freq_offset = rf_gz.amplitude * delta_z * (z + 0.5) + freq_offset[m]
                  
                  #Add rf block
                  seq.add_block(rf_ev, rf_gz, slc_lbl)
                  seq.met_block_id.append(m)
         
               #Prephasing block setup
               if no_pe == 1:
                  gy_pre_ev[m].amplitude = 0
                  if readout == 0:
                     gy_up_ev[m].waveform *= 0
                     gy_dn_ev[m].waveform *= 0
                     gy_ev[m].waveform *= 0
                  else:
                     gy_ev[m].amplitude = 0
                  if acq_3d == 1 and n_z[m] > 1:
                     gz_pre_ev[m].amplitude = 0
               elif acq_3d == 1 and n_z[m] > 1:
                  pre_amp = gz_pre_amp[m] * (1 - z * 2 / (n_z[m]))
                  gz_pre_ev[m].amplitude = pre_amp * slice_on
            
               #Add prephasing block
               if acq_3d == 0 or n_z[m] == 1:
                  if rf_types[m] == 0:
                     seq.add_block(gx_pre_ev[m], gy_pre_ev[m], slc_lbl)
                  else:
                     seq.add_block(gx_pre_ev[m], gy_pre_ev[m], rf_gz_r, slc_lbl)
               else:
                  if rf_types[m] == 0:
                     seq.add_block(pp.make_delay(10E-6))
                     seq.met_block_id.append(m)
                     seq.add_block(gx_pre_ev[m], gy_pre_ev[m], gz_pre_ev[m], slc_lbl)
                  else:
                     seq.add_block(gx_pre_ev[m], gy_pre_ev[m], rf_gz_r, gz_pre_ev[m], slc_lbl)
               seq.met_block_id.append(m)
               
               #Loop through phase encodes
               for i in range(np.int32(n_y_acq[m]) + skip_acq):
         
                  #Flip readout polarity if necessary
                  if readout == 0 and i % 2 == 1:
                     gx_ev[m].amplitude = gx_amp * -1
                  else:
                     gx_ev[m].amplitude = gx_amp

                  #Add data acquisition and readout gradient
                  if readout == 0 and n_y_acq[m] > 1:
                     if i == n_y_acq[m] - 1 + skip_acq:
                        seq.add_block(adc_ev[m], gx_ev[m], gy_dn_ev[m], slc_lbl)   
                     elif i == 0:
                        seq.add_block(adc_ev[m], gx_ev[m], gy_up_ev[m], slc_lbl)
                     elif skip_acq == 1 and gy_pre_ev[m].area + 1 / fov[1] * i == 0:
                        seq.add_block(adc_ev[m], gx_ev[m], gy_dn_ev[m], slc_lbl)
                     elif skip_acq == 1 and gy_pre_ev[m].area + 1 / fov[1] * (i - 1) == 0:
                         seq.add_block(adc_ev[m], gx_ev[m], gy_up_ev[m], slc_lbl)
                     else:
                        seq.add_block(adc_ev[m], gx_ev[m], gy_ev[m], slc_lbl)
                  else:
                     seq.add_block(adc_ev[m], gx_ev[m], slc_lbl)
                  seq.met_block_id.append(m)
                  
                  #Add flack gradient and phase blip if necessary
                  if i != n_y_acq[m] - 1 and readout == 1:
                     seq.add_block(gy_ev[m], fly_ev[m], slc_lbl)
                     seq.met_block_id.append(m)
                     
               #Spoilers if requested
               if grad_spoil == 1:
                  x_spoil_ev[m].amplitude = spoil_amp[m] * np.sign(gx_ev[m].amplitude)
                  y_spoil_ev[m].amplitude = spoil_amp[m] * np.sign(gy_ev[m].amplitude)
                  if acq_3d == 1:
                     z_spoil_ev[m].amplitude = -gz_pre_ev[m].amplitude
                  seq.add_block(x_spoil_ev[m], y_spoil_ev[m], z_spoil_ev[m], slc_lbl)
                  seq.met_block_id.append(m)
                  
               #Add TR delay
               exc_dur = np.sum(seq.block_durations[block_start::])
               if tr > 0:
                  tr_delay = tr - exc_dur
                  seq.add_block(pp.make_delay(tr_delay))
                  seq.met_block_id.append(m)
                  tv_delay -= tr
                  ts_delay -= tr
               else:
                  tv_delay -= exc_dur
                  ts_delay -= exc_dur
      
            #Add volume delay
            if tv > 0:
               seq.add_block(pp.make_delay(tv_delay))
               seq.met_block_id.append(m)
               ts_delay -= tv_delay
      
         #Add metabolite set delay
         if ts > 0:
            seq.add_block(pp.make_delay(ts_delay))
            seq.met_block_id.append(m)
      
         if a == 0 and r == 0 and return_plot is True:
            plot_seq = deepcopy(seq)
            
      #Add spectra if necessary
      if win.check_spectra_end.isChecked():
         seq.add_block(spec_pulse, spec_gz, end_lbl, slc_lbl)
         seq.met_block_id.append(m)
         seq.add_block(spec_gz_r, end_lbl, slc_lbl)
         seq.met_block_id.append(m)
         seq.add_block(spec_adc, end_lbl)
         seq.met_block_id.append(m)

   if return_plot is True:
      return seq, plot_seq
   else:
      return seq
       
def compute_k_space(seq, n_met):

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
   rf_i = interp.interp1d(np.real(wave_data[3][0, :]), wave_data[3][1, :], fill_value=0.0, 
                         bounds_error=False)(t_i)                       

   #Find right edges of rf/blocks
   rf_edges = [[] for i in range(n_met)]
   rf_durs = [[] for i in range(n_met)]
   block_edges = np.cumsum(seq.block_durations)                    
   for i in range(1, len(seq.block_events) + 1):
      rf_num = seq.block_events[i][1]
      met_idx = seq.met_block_id[i - 1]
      if rf_num > 0 and type(met_idx) is int:
         rf_edges[met_idx].append(block_edges[i - 1])
         rf_durs[met_idx].append(seq.block_durations[i - 1])
  
   #Make empty arrays for storing data
   k_list = []
   adc_list = []
   k_2d_list = []
   adc_2d_list = []
   g_list = []
   rf_list = []
         
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
      
   return k_list, adc_list, k_2d_list, adc_2d_list, g_list, rf_list            
