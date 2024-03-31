"""
Testing for x_epi class
"""

import filecmp
import json
from os.path import abspath, basename, dirname, join, splitext
from os import remove
import unittest
from jsoncomparison import Compare
import nibabel as nib
import numpy as np
from x_epi.data import XData
from x_epi.utils import BASE_DIR

FIX_DIR = abspath(join(BASE_DIR, "..", "tests/fixtures"))

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as jid:
        param_dic = json.load(jid)
    return param_dic

def extract_common_pars(seq_dic, update_dic=None):
    
    # Update sequence dictionary with new values
    if update_dic is not None:
        for key, val in update_dic.items():
            seq_dic[key] = val
    
    # Get parameters common to all metabolites 
    return {key: seq_dic[key] for key in seq_dic if key != "mets"}
    
def load_data(seq_dic, twix_path, update_dic=None, recon_dims=True, coord_path=None):
    
    #Extract parameters common to all metabolites
    cmn_dic = extract_common_pars(seq_dic, update_dic=update_dic)

    # Create data class
    x_data = XData(**cmn_dic)
    for i in range(cmn_dic['n_met']):
        x_data.add_met(**seq_dic["mets"][i])

    # Read in the data
    x_data.load_k_data(twix_path, recon_dims=recon_dims)
    
    # Get coordinates if necessary
    if coord_path is not None:
        x_data.load_k_coords(coord_path)
        x_data.regrid_k_data(method='nufft')
    
    # Flip axes
    x_data.flip_k_data()
    
    return x_data
    
def save_test_output(x_data, mean=False):
    x_data.save_nii(f'{FIX_DIR}/test')
    if mean is True:
        x_data.save_nii(f'{FIX_DIR}/test', mean=True)
    x_data.save_param_dic(f'{FIX_DIR}/test_params')


class TestXData(unittest.TestCase):

    def compare_imgs(self, img_list, ref_root):
        for img in img_list:
            fix_img = nib.load(f'{FIX_DIR}/{ref_root}_{img}.nii.gz').get_fdata()
            test_img = nib.load(f'{FIX_DIR}/test_{img}.nii.gz').get_fdata()
            comp_img = np.allclose(fix_img, test_img, rtol=1E-4) 
            self.assertTrue(comp_img, msg=f"{ref_root} {img} test failed")
            remove(f'{FIX_DIR}/test_{img}.nii.gz')

    def compare_json(self, ref_root):
        ref_dic = load_json(f'{FIX_DIR}/{ref_root}_params.json')
        test_dic = load_json(f'{FIX_DIR}/test_params.json')
        json_comp = Compare().check(test_dic, ref_dic)
        self.assertEqual(json_comp, {}, "Should return an empty dictionary")
        remove(f'{FIX_DIR}/test_params.json')

    def test_proton_recon(self):
        
        # Load in json data describing data/sequence
        json_path = f'{FIX_DIR}/phantom.json'
        param_dic = load_json(json_path)
        
        # Get k-space data
        new_pars = {'n_rep':5, 'ts':1, 'n_chan':32}
        x_data = load_data(
            param_dic, f'{FIX_DIR}/phantom.dat', update_dic=new_pars
        )

        # And the same for the reference data
        x_ref = load_data(
            param_dic, f'{FIX_DIR}/phantom_ref.dat', update_dic=new_pars
        )

        # Run fft recon
        x_data.fft_recon(ref_data=x_ref)
        x_data.combine_coils()

        # Run tests
        save_test_output(x_data, mean=True)
        self.compare_imgs(['proton', 'proton_mean'], 'phantom')
        self.compare_json('phantom')

    def test_ramp_recon(self):
        
        # Load in json data describing data/sequence
        json_path = f'{FIX_DIR}/ramp_samp.json'
        json_ref_path = f'{FIX_DIR}/ramp_samp_ref.json'
        param_dic = load_json(json_path)
        param_ref_dic = load_json(json_ref_path)
        
        # Get k-space data
        new_pars = {'n_avg':32}
        coord_path = f'{FIX_DIR}/ramp_samp_k_data.npy'
        x_data = load_data(
            param_dic,
            f'{FIX_DIR}/ramp_samp.dat',
            update_dic=new_pars,
            recon_dims=False,
            coord_path=coord_path
        )

        # And the same for the reference data
        x_ref = load_data(
            param_ref_dic,
            f'{FIX_DIR}/ramp_samp_ref.dat',
            recon_dims=False,
            coord_path=coord_path
        )

        # Run fft recon
        x_data.fft_recon(ref_data=x_ref, point=False)
        x_data.apply_off_res()
        x_data.combine_coils()
        
        # Run tests
        save_test_output(x_data)
        self.compare_imgs(['eg', 'me'], 'ramp_samp')
        self.compare_json('ramp_samp')

    def test_3d_recon(self):
    
        # Load in json data describing data/sequence
        json_path = f'{FIX_DIR}/epi_3d.json'
        json_ref_path = f'{FIX_DIR}/epi_3d_ref.json'
        param_dic = load_json(json_path)
        param_ref_dic = load_json(json_ref_path)
        
        # Get k-space data
        new_pars = {'n_rep':20}
        x_data = load_data(
            param_dic,  f'{FIX_DIR}/epi_3d.dat', update_dic=new_pars
        )
        
        # And the same for the reference data
        x_ref = load_data(param_ref_dic, f'{FIX_DIR}/epi_3d_ref.dat')

        # Run fft recon
        x_data.fft_recon(ref_data=x_ref, point=False)
        x_data.apply_off_res()
        x_data.apply_phase_shift()
        x_data.combine_coils()

        # Run tests
        save_test_output(x_data)
        self.compare_imgs(['pyr', 'lac'], 'epi_3d')
        self.compare_json('epi_3d')
                
        
if __name__ == "__main__":
    unittest.main()
