"""
Testing for x_epi utility funtions
"""

from os.path import join
from os import remove
import pickle
import unittest
import numpy as np
from x_epi.utils import (
    nuc_to_gamma,
    load_ssrf_grad,
    load_ssrf_rf,
    interp_waveform,
    compute_k_space,
    save_k_space,
    BASE_DIR,
    RES_DIR,
)

FIX_DIR = join(BASE_DIR, "..", "tests/fixtures")


class TestNucToGamma(unittest.TestCase):
    def test_nuc_to_gamma_known_str(self):
        self.assertEqual(nuc_to_gamma("13C"), 10.7084e6, "Should be 10.7084E6")

    def test_nuc_to_gamma_unknown_str(self):
        with self.assertRaises(KeyError):
            nuc_to_gamma("14C")

    def test_nuc_to_gamma_num(self):
        with self.assertRaises(KeyError):
            nuc_to_gamma(1)


class TestLoadSsrfGrad(unittest.TestCase):
    def test_load_ssrf_grad_output(self):
        # Load in gradient info
        outputs = load_ssrf_grad(join(RES_DIR, "siemens_singleband_pyr_3T.GRD"))

        # Test number of args
        self.assertEqual(len(outputs), 3, "Should be 3")

        # Test output types
        self.assertIs(type(outputs[0]), np.ndarray)
        self.assertIs(type(outputs[1]), float)
        self.assertIs(type(outputs[2]), float)

    def test_load_ssrf_grad_no_file(self):
        with self.assertRaises(FileNotFoundError):
            load_ssrf_grad("123.GRD")

    def test_load_ssrf_grad_bad_file(self):
        with self.assertRaises(TypeError):
            load_ssrf_grad(join(RES_DIR, "siemens_singleband_pyr_3T.RF"))


class TestLoadSsrfRf(unittest.TestCase):
    def test_load_ssrf_rf_output(self):
        # Load in gradient info
        outputs = load_ssrf_rf(join(RES_DIR, "siemens_singleband_pyr_3T.RF"))

        # Test number of args
        self.assertEqual(len(outputs), 4, "Should be 4")

        # Test output types
        self.assertIs(type(outputs[0]), np.ndarray)
        self.assertIs(type(outputs[1]), np.ndarray)
        self.assertIs(type(outputs[2]), float)
        self.assertIs(type(outputs[3]), float)

    def test_load_ssrf_rf_no_file(self):
        with self.assertRaises(FileNotFoundError):
            load_ssrf_rf("123.RF")

    def test_load_ssrf_rf_bad_file(self):
        with self.assertRaises(TypeError):
            load_ssrf_rf(join(RES_DIR, "siemens_singleband_pyr_3T.GRD"))


class TestInterpWaveform(unittest.TestCase):
    def test_interp_waveform_output(self):
        # Generate fake testing data
        sig = np.arange(20)
        delta_t = 1
        delta_ti = 0.5
        output = interp_waveform(sig, delta_t, delta_ti)

        # Check the number of output points
        self.assertEqual(output.shape[0], 40, "Should be 40")

        # Check the interpolation itself
        self.assertEqual(output[2], 0.5, "Should be 0.5")

        # Check output with ti_end option
        output = interp_waveform(sig, delta_t, delta_ti, ti_end=20)
        self.assertEqual(output[-1], 19, "Should be 19")


class TestComputeKSpace(unittest.TestCase):
    def test_c_no_z(self):
        # Load in assets for testing
        with open(join(FIX_DIR, "k_space_no_z.pkl"), "rb") as f_id:
            seq = pickle.load(f_id)
        output_ref = np.load(join(FIX_DIR, "k_space_no_z.npy"), allow_pickle=True)

        # Get k-space
        output = compute_k_space(seq)

        # Check number of outputs
        self.assertEqual(len(output), 6, "Should be 6")

        # Check that third dimension is all zeros
        z_unq = np.unique(output[0][0][2, :])
        self.assertEqual(z_unq.shape[0], 1, "Should be 1")
        self.assertEqual(z_unq[0], 0, "Should be 0")

        # Check that k-space data matches
        self.assertTrue(np.allclose(output[0][0], output_ref[0][0]))

    def test_compute_k_space_3d(self):
        # Load in assets for testing
        with open(join(FIX_DIR, "k_space_3d.pkl"), "rb") as f_id:
            seq = pickle.load(f_id)
        output_ref = np.load(join(FIX_DIR, "k_space_3d.npy"), allow_pickle=True)

        # Get k-space
        output = compute_k_space(seq)

        # Check number of outputs
        self.assertEqual(len(output), 6, "Should be 6")

        # Check that k-space data matches
        self.assertTrue(np.allclose(output[0][0], output_ref[0][0]))


class TestSaveKSpace(unittest.TestCase):
    def test_save_k_space_output(self):
        # Load in assets for testing
        with open(join(FIX_DIR, "k_space_3d.pkl"), "rb") as f_id:
            seq = pickle.load(f_id)
        output_ref = np.load(join(FIX_DIR, "k_space_3d.npy"), allow_pickle=True)

        # Save k-space to temporary file
        k_out_root = join(FIX_DIR, "test")
        save_k_space(seq, k_out_root)

        # Load in saved k-space data and check it
        k_out_path = k_out_root + "_kspace.npy"
        output = np.load(k_out_path, allow_pickle=True)
        for i in range(output.shape[0]):
            self.assertTrue(np.allclose(output[i], output_ref[1][i]))

        # Delete test file
        remove(k_out_path)


if __name__ == "__main__":
    unittest.main()
