#!/usr/bin/python3

#Load libs
from os.path import join
import pickle
import numpy as np
import x_epi
from x_epi.utils import *
from x_epi.XEPI import XEPI

FIX_DIR = os.path.dirname(__file__)

###########################
### For 2D k-space test ###
###########################

#Make sequence
seq = XEPI(acq_3d=False, no_slc=True, slice_axis='X')
seq.add_met(size=[16, 16, 1], use_sinc=True)
seq.add_met(size=[16, 16, 1], use_sinc=True)
seq.create_seq()

#Save object
with open(join(FIX_DIR, 'k_space_no_z.pkl'), 'wb') as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

#Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, 'k_space_no_z.npy'), np.array(k_space, dtype=object))

###########################
### For 3D k-space test ###
###########################

seq = XEPI(acq_3d=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.create_seq()

#Save object
with open(join(FIX_DIR, 'k_space_3d.pkl'), 'wb') as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

#Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, 'k_space_3d.npy'), np.array(k_space, dtype=object))

###########################
### For 3D k-space test ###
###########################

seq = XEPI(acq_3d=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.create_seq()

#Save object
with open(join(FIX_DIR, 'k_space_3d.pkl'), 'wb') as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

#Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, 'k_space_3d.npy'), np.array(k_space, dtype=object))

#######################
###  Sequence tests ###
#######################

#3d sequence, partial fouriuer in z, multi-echo, two time points, alternating directions
seq_1 = XEPI(grad_spoil=True, n_rep=2, alt_read=True, alt_pha=True,
                 alt_slc=True, n_echo=3, tv=1000, ts=2500)
seq_1.add_met(pf_pe2=0.75, z_centric=True)
seq_1.add_met(use_sinc=True)
_ = seq_1.create_seq(return_plot=True)
seq_1.write(join(FIX_DIR, 'seq_1.seq'))
seq_1.save_params(join(FIX_DIR, 'seq_1'))

#2d sequence, ramp sampling, flyback, partial fourier in y
seq_2 = XEPI(ramp_samp=True, max_slew=120, max_grad=60, acq_3d=False,
                 symm_ro=False, grad_spoil=True, n_echo=3, tr=500,
                 delta_te=150)
seq_2.add_met(ro_os=2, pf_pe=0.75, use_sinc=True)
seq_2.add_met(use_sinc=False, freq_off=50)
seq_2.add_spec(run_spec='BOTH')
seq_2.create_seq()
seq_2.write(join(FIX_DIR, 'seq_2.seq'))
seq_2.save_params(join(FIX_DIR, 'seq_2'))

#No phase encoding gradients, symmetric, 2D
seq_3 = XEPI(no_pe=True, no_slc=True, n_reps=2)
seq_3.add_met()
seq_3.create_seq(no_reps=True)
seq_3.write(join(FIX_DIR, 'seq_3.seq'))
seq_3.save_params(join(FIX_DIR, 'seq_3'))

#No phase encoding, flyback, 3D
seq_4 = XEPI(no_pe=True, acq_3d=True, symm_ro=False)
seq_4.add_met()
seq_4.create_seq()
seq_4.write(join(FIX_DIR, 'seq_4.seq'))
seq_4.save_params(join(FIX_DIR, 'seq_4'))
