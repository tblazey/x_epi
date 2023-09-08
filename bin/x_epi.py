#!/usr/bin/python

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from x_epi import * 

#Custom range checker for argparse
def range_wrapper(v_min, v_max):
   def range_check(string):
      value = float(string)
      if value < v_min or value > v_max:
         msg = f"{value} is not within {v_min} to {v_max}"
         raise argparse.ArgumentTypeError(msg)
      return value
   return range_check

#Argument parser that are common to each metabolite
main_parser = argparse.ArgumentParser(add_help=False)
main_parser.add_argument('-out', help='Root for file outputs', default='epi')

#Image parameters
img_grp = main_parser.add_argument_group('Image parameters')
img_grp.add_argument('-fov', type=float, nargs=3, metavar=('RO', 'PE', 'SLC'),
                      help='Field of view in readout, phase, and slice dimensions [mm]',
                      default=[240, 240, 240])
img_grp.add_argument('-rbw', type=float, help='Readout bandwidth [kHz]', default=50,
                     metavar='BW')
img_grp.add_argument('-n_avg', type=int, help='Number of averages', metavar='N',
                      default=1)
img_grp.add_argument('-n_rep', type=int, help='Number of repetitions', metavar='N',
                      default=1)                                       
img_grp.add_argument('-tr', type=float, help='Minimum repetition time [ms]', default=0)
img_grp.add_argument('-tv', type=float, help='Minimum volume time [ms]', default=0)
img_grp.add_argument('-ts', type=float, help='Minimum set time [ms]', default=0)
img_grp.add_argument('-ro_off', type=float, metavar='OFFSET',
                     help='Offset in readout dimension [mm]', default=0)  
img_grp.add_argument('-pe_off', type=float, metavar='OFFSET',
                     help='Offset in phase encoding dimension [mm]', default=0)                           
img_grp.add_argument('-slc_off', type=float, metavar='OFFSET', default=0,
                     help='Offset in slice/second phase encoding dimension [mm]')                    
img_grp.add_argument('-n_echo', type=int, help='Number of echos', metavar='N', default=1)
img_grp.add_argument('-delta_te', type=float, help='Minimum time between echoes [ms]',
                     metavar='DELTA', default=0)
img_grp.add_argument('-ori', choices=['Sagittal', 'Coronal', 'Transverse'],
                     default='Transverse',
                     help='Acquistion orientation. Only used for reconstruction')
img_grp.add_argument('-pe_dir', choices=['AP', 'RL', 'SI'], default='AP',
                     help='Phase encoding direction. Only used for reconstruction')
                      
#Acquisition options
acq_grp = main_parser.add_argument_group('Acquisition options')
acq_grp.add_argument('-symm_ro', action='store_true',
                     help='Replace flyback readout with a symmetric one.')
acq_grp.add_argument('-acq_3d', action='store_true', help='Use a 3D readout instead of 2D') 
acq_grp.add_argument('-no_pe', action='store_true', help='Turn phase encoding on/off')
acq_grp.add_argument('-no_slc', action='store_true', help='Turn off slice gradients')
acq_grp.add_argument('-grad_spoil', action='store_true', help='Turn gradient spoiling on')
acq_grp.add_argument('-ramp_samp', action='store_true', help='Turn on ramp sampling')   
acq_grp.add_argument('-ro_os', default=1.0, type=float, metavar='FACTOR',
                     help='Readout oversampling factor for ramp sampling')                                                                                                         
acq_grp.add_argument('-slice_axis', choices=['X', 'Y', 'Z'], default='Z', type=str,
                     help='Axis to play slice selection gradients')
acq_grp.add_argument('-alt_read', action='store_true',
                     help='Alternate polarity of readout gradient every repeition')                             
acq_grp.add_argument('-alt_pha', action='store_true',
                     help='Alternate polarity of phase encoding gradient every repeition') 
acq_grp.add_argument('-alt_slc', action='store_true',
                     help='Alternate polarity of second phase encoding every repetition')
                     
#Spectra options   
spec_grp = main_parser.add_argument_group('Spectra options')
spec_grp.add_argument('-run_spec', choices=['START', 'END', 'BOTH', 'NO'], default='NO',
                     help='Run a slice selective spectra at the start and/or ' \
                          'end of the seqience.', type=str)
spec_grp.add_argument('-spec_size', type=int, default=2048, metavar='N',
                     help='Number of points in spectra') 
spec_grp.add_argument('-spec_bw', type=float, default=25, metavar='BW',
                      help='Spectral bandwidth [kHz]')
spec_grp.add_argument('-spec_flip', type=float, default=2.5, metavar='ANGLE',
                      help='Flip angle for slcie selective pulse [deg]')
spec_grp.add_argument('-spec_tr', type=float, default=1000, metavar='TR',
                      help='Minimum repetition time for spectra acquistions [ms]')                                            
spec_grp.add_argument('-n_spec', type=int, default=1, metavar='N',
                     help='Number of spectra acquisitions')                     
#Scanner options                                          
scan_grp = main_parser.add_argument_group('Scanner Options')
scan_grp.add_argument('-b0', type=range_wrapper(0.1, 20), default=3,
                      help='B0 Field Strength [Tesla]')  
scan_grp.add_argument('-nuc', choices=['13C', '1H', '2H', '15N', '17O', '31P', '19F',
                                       '23Na', '129Xe'],
                      help='Nucleus to imaage', default='13C', type=str)  
scan_grp.add_argument('-max_grad', help='Maximum gradient strength [mT/m]', default=100,
                      type=float, metavar='GRAD')
scan_grp.add_argument('-max_slew', help='Maximum slew rate [T/m/s]', default=200,
                      type=float, metavar='SLEW')   
scan_grp.add_argument('-rf_ringdown_time', help='RF ringdown time [us]', default=30,
                      type=float)
scan_grp.add_argument('-rf_dead_time', help='RF dead time [us]', default=100,
                      type=float)                       
                                                                 
#Metabolite specific parser
met_parser = argparse.ArgumentParser(description='x_epi: An EPI sequence for X-Nucleus' \
                                                 ' imaging', 
                                     parents=[main_parser],
                                     epilog='Each metabolite must be preceded by -met.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
met_parser.add_argument('-name', help='Metabolite name')

#Parameters describing metabolite specific waveforms
wav_grp = met_parser.add_argument_group('Metabolite Specific Waveforms')
wav_grp.add_argument('-grd_path', type=str, metavar='PATH',
                     default=f'{XEPI.ssrf_dir}/siemens_singleband_pyr_3T.GRD',
                     help='Path to gradient file from Spectral-Spatial RF Toolbox')                     
wav_grp.add_argument('-rf_path', type=str, metavar='PATH',
                     default=f'{XEPI.ssrf_dir}/siemens_singleband_pyr_3T.RF',
                     help='Path to RF file from Spectral-Spatial RF Toolbox')
wav_grp.add_argument('-formula', type=str, default='1',
                     help='Formula for scaling SSRF gradient where x is slice' \
                          ' thickness in mm')
                                               
#Parameters describing metabolite specific excitation
exc_grp = met_parser.add_argument_group('Metabolite Specific Excitation')
exc_grp.add_argument('-use_sinc', action='store_true',
                     help='Use sinc pulse instead of an ssrf pulse')
exc_grp.add_argument('-flip', default=90, type=float, help='Flip angle [deg]',
                     metavar='ANGLE')
exc_grp.add_argument('-freq_off', type=float, default=0,
                     help='Frequency offset [Hz]', metavar='OFFSET')
exc_grp.add_argument('-sinc_dur', default=4, type=float, metavar='DUR',
                     help='Duration of sinc pulse [ms]')                     
exc_grp.add_argument('-sinc_frac', default=0.5, type=float, metavar='FRAC',
                     help='Center fraction of sinc pulse')     
exc_grp.add_argument('-sinc_tbw', default=4, type=float, metavar='TBW',
                     help='Time-bandwidth product of sinc pulse') 
                        
#Image parameters specific to each metabolite
met_grp = met_parser.add_argument_group('Metabolite Specific Imaging Parameters')            
met_grp.add_argument('-size', type=int, nargs=3, metavar=('RO', 'PE', 'SLC'),
                     help='Grid size in readout, phase, and slice dimensions',
                     default=[16, 16, 16])
met_grp.add_argument('-pf_pe', type=range_wrapper(0.5, 1.0), default=1, metavar='FRAC',
                     help='Partial Fourier in phase encoding dimension')   
met_grp.add_argument('-pf_pe2', type=range_wrapper(0.5, 1.0), default=1, metavar='FRAC',
                     help='Partial Fourier in second phase encoding dimension')
met_grp.add_argument('-z_centric', action='store_true',
                     help='Use centric encoding instead of linear for second phase' \
                          ' encoding dimension.')                 

#Parse main arguments and return a list of unparsed metabolite specific arguments
img_args, met_up_args = main_parser.parse_known_args()

seq = XEPI(**vars(img_args))
if img_args.run_spec != "NO":
   seq.add_spec(**vars(img_args))

#Parse metabolite specific options if necessary
met_args = []
if len(met_up_args) > 0:
   
   #Show help if needed
   if '-h' in met_up_args:
      met_parser.print_help()
      sys.exit()
   
   #Make sure first argument is metabolite flag
   if met_up_args[0] != '-met':
      raise met_parser.error('Must specify a -met flag when modifying ' \
                             'metabolite parameters')
      
   #Split args so we get a separate argument namespace for each metabolite
   met_arg_list = []
   for idx, arg in enumerate(met_up_args[1::]):
      if arg == '-met':
         seq.add_met(**vars(met_parser.parse_args(met_arg_list)))
         met_arg_list = []
      else:
         met_arg_list.append(arg)
      if idx == len(met_up_args) - 2:
         seq.add_met(**vars(met_parser.parse_args(met_arg_list)))

else:

   #If no metabolites are specified, add one to the list
   seq.add_met(**vars(met_parser.parse_args(['-name', 'default'])))

#Create sequence
plot_seq = seq.create_seq(return_plot=True)

#Save k-space data
save_k_space(plot_seq, img_args.out)

#Save sequence
seq.write(img_args.out)

#Save parameters
seq.save_params(img_args.out)

