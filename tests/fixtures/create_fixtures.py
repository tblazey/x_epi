#!/usr/bin/python3

# Load libs
import json
from os.path import join
import pickle
import numpy as np
import x_epi
from x_epi.utils import *
from x_epi.data import XData
from x_epi.seq import XSeq


FIX_DIR = os.path.dirname(__file__)

###########################
### For 2D k-space test ###
###########################

# Make sequence
seq = XSeq(acq_3d=False, no_slc=True, slice_axis="X")
seq.add_met(size=[16, 16, 1], use_sinc=True)
seq.add_met(size=[16, 16, 1], use_sinc=True)
seq.create_seq()

# Save object
with open(join(FIX_DIR, "k_space_no_z.pkl"), "wb") as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

# Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, "k_space_no_z.npy"), np.array(k_space, dtype=object))

###########################
### For 3D k-space test ###
###########################

seq = XSeq(acq_3d=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.create_seq()

# Save object
with open(join(FIX_DIR, "k_space_3d.pkl"), "wb") as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

# Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, "k_space_3d.npy"), np.array(k_space, dtype=object))

###########################
### For 3D k-space test ###
###########################

seq = XSeq(acq_3d=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.add_met(size=[16, 16, 4], use_sinc=True)
seq.create_seq()

# Save object
with open(join(FIX_DIR, "k_space_3d.pkl"), "wb") as f_id:
    pickle.dump(seq, f_id, pickle.HIGHEST_PROTOCOL)

# Save k-space
k_space = compute_k_space(seq)
np.save(join(FIX_DIR, "k_space_3d.npy"), np.array(k_space, dtype=object))

#######################
###  Sequence tests ###
#######################

# 3d sequence, partial fouriuer in z, multi-echo, two time points, alternating directions
seq_1 = XSeq(
    grad_spoil=True,
    n_rep=2,
    alt_read=True,
    alt_pha=True,
    alt_slc=True,
    n_echo=3,
    tv=1000,
    ts=2500,
)
seq_1.add_met(pf_pe2=0.75, z_centric=True)
seq_1.add_met(use_sinc=True)
_ = seq_1.create_seq(return_plot=True)
seq_1.write(join(FIX_DIR, "seq_1.seq"))
seq_1.save_params(join(FIX_DIR, "seq_1"))

# 2d sequence, ramp sampling, flyback, partial fourier in y
seq_2 = XSeq(
    ramp_samp=True,
    max_slew=120,
    max_grad=60,
    acq_3d=False,
    symm_ro=False,
    grad_spoil=True,
    n_echo=3,
    tr=500,
    delta_te=150,
    ro_os=2.0,
)
seq_2.add_met(pf_pe=0.75, use_sinc=True)
seq_2.add_met(use_sinc=False, freq_off=50)
seq_2.add_spec(run_spec="BOTH")
seq_2.create_seq()
seq_2.write(join(FIX_DIR, "seq_2.seq"))
seq_2.save_params(join(FIX_DIR, "seq_2"))

# No phase encoding gradients, symmetric, 2D
seq_3 = XSeq(no_pe=True, no_slc=True)
seq_3.add_met()
seq_3.create_seq(no_reps=True)
seq_3.write(join(FIX_DIR, "seq_3.seq"))
seq_3.save_params(join(FIX_DIR, "seq_3"))

# No phase encoding, flyback, 3D
seq_4 = XSeq(no_pe=True, acq_3d=True, symm_ro=False)
seq_4.add_met()
seq_4.create_seq()
seq_4.write(join(FIX_DIR, "seq_4.seq"))
seq_4.save_params(join(FIX_DIR, "seq_4"))


########################
###  XData Ramp Test ###
########################

# Load in json data describing data/sequence
json_path = f'{FIX_DIR}/ramp_samp.json'
with open(json_path, "r", encoding="utf-8") as jid:
    param_dic = json.load(jid)
    
# Do the same thing for reference
json_ref_path = f'{FIX_DIR}/ramp_samp_ref.json'
with open(json_ref_path, "r", encoding="utf-8") as jid:
    param_ref_dic = json.load(jid)

# Extract parameters common to all metabolites
param_dic["n_avg"] = 32
seq_dic = {key: param_dic[key] for key in param_dic if key != "mets"}
seq_ref_dic = {key: param_ref_dic[key] for key in param_ref_dic if key != "mets"}

# Create class for recon
x_data = XData(**seq_dic)
x_data.add_met(**param_dic["mets"][0])
x_data.add_met(**param_dic["mets"][1])

# Load in k-space data
twix_path = f'{FIX_DIR}/ramp_samp.dat'
coord_path = f'{FIX_DIR}/ramp_samp_k_data.npy'
x_data.load_k_data(twix_path, recon_dims=False)
x_data.load_k_coords(coord_path)
x_data.regrid_k_data(method='nufft')
x_data.flip_k_data()

# Create class for reference data
x_ref = XData(**seq_ref_dic)
x_ref.add_met(**param_ref_dic["mets"][0])
x_ref.add_met(**param_ref_dic["mets"][1])

# Load in reference k-space data
ref_path = f'{FIX_DIR}/ramp_samp_ref.dat'
x_ref.load_k_data(ref_path, recon_dims=False)
x_ref.load_k_coords(coord_path)
x_ref.regrid_k_data(method='nufft')
x_ref.flip_k_data()

# Run fft recon
x_data.fft_recon(ref_data=x_ref, point=False)
x_data.apply_off_res()
x_data.combine_coils()

# Save output
x_data.save_nii(f'{FIX_DIR}/ramp_samp')
test_json_path = f'{FIX_DIR}/ramp_samp_params'
x_data.save_param_dic(test_json_path)

######################
###  XData 3D Test ###
######################

# Load in json data describing data/sequence
json_path = f'{FIX_DIR}/epi_3d.json'
with open(json_path, "r", encoding="utf-8") as jid:
    param_dic = json.load(jid)
    
# Do the same thing for reference
json_ref_path = f'{FIX_DIR}/epi_3d_ref.json'
with open(json_ref_path, "r", encoding="utf-8") as jid:
    param_ref_dic = json.load(jid)

# Extract parameters common to all metabolites
param_dic["n_rep"] = 20
seq_dic = {key: param_dic[key] for key in param_dic if key != "mets"}
seq_ref_dic = {key: param_ref_dic[key] for key in param_ref_dic if key != "mets"}

# Create class for recon
x_data = XData(**seq_dic)
x_data.add_met(**param_dic["mets"][0])
x_data.add_met(**param_dic["mets"][1])

# Load in k-space data
twix_path = f'{FIX_DIR}/epi_3d.dat'
x_data.load_k_data(twix_path)
x_data.flip_k_data()

# Create class for reference data
x_ref = XData(**seq_ref_dic)
x_ref.add_met(**param_ref_dic["mets"][0])
x_ref.add_met(**param_ref_dic["mets"][1])

# Load in reference k-space data
ref_path = f'{FIX_DIR}/epi_3d_ref.dat'
x_ref.load_k_data(ref_path)
x_ref.flip_k_data()

# Run fft recon
x_data.fft_recon(ref_data=x_ref, point=False)
x_data.apply_off_res()
x_data.apply_phase_shift()
x_data.combine_coils()

# Save output
x_data.save_nii(f'{FIX_DIR}/epi_3d')
x_data.save_param_dic(f'{FIX_DIR}/epi_3d_params')

##########################
###  XData Proton Test ###
##########################

# Load in json data describing data/sequence
json_path = f'{FIX_DIR}/phantom.json'
with open(json_path, "r", encoding="utf-8") as jid:
    param_dic = json.load(jid)

# Extract parameters common to all metabolites
param_dic["n_rep"] = 5
param_dic["ts"] = 1
param_dic["n_chan"] = 32
seq_dic = {key: param_dic[key] for key in param_dic if key != "mets"}

# Create class for recon
x_data = XData(**seq_dic)
x_data.add_met(**param_dic["mets"][0])

# Load in k-space data
twix_path = f'{FIX_DIR}/phantom.dat'
x_data.load_k_data(twix_path)
x_data.flip_k_data()

# Create class for reference data
x_ref = XData(**seq_dic)
x_ref.add_met(**param_dic["mets"][0])

# Load in reference k-space data
ref_path = f'{FIX_DIR}/phantom_ref.dat'
x_ref.load_k_data(ref_path)
x_ref.flip_k_data()

# Run fft recon
x_data.fft_recon(ref_data=x_ref)
x_data.combine_coils()

# Save output
x_data.save_nii('phantom')
x_data.save_nii('phantom', mean=True)
x_data.save_param_dic('phantom_params')


