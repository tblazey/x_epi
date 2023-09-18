"""
Testing for x_epi command line utility
"""

# Load libs
import argparse
import filecmp
import json
from os.path import abspath, dirname, join
from os import remove
import sys
import unittest
import numpy as np
from x_epi.bin.x_epi_cmd import range_wrapper, main
from x_epi.utils import BASE_DIR

sys.path.insert(0, dirname(__file__))
from test_x_seq import comp_json

FIX_DIR = abspath(join(BASE_DIR, "..", "tests/fixtures"))


def create_arg_str(arg_dic):
    arg_str = ""
    for key, value in arg_dic.items():
        if value is True:
            arg_str += f" -{key}"
        elif value is not False:
            arg_str += f" -{key} {value}"
    return arg_str


class TestRangeWrapper(unittest.TestCase):
    def test_range_wrapper_outside(self):
        range_check = range_wrapper(1, 10)
        with self.assertRaises(argparse.ArgumentTypeError):
            range_check(str(-1))
        with self.assertRaises(argparse.ArgumentTypeError):
            range_check(str(11))

    def test_range_wrapper_inside(self):
        range_check = range_wrapper(1, 10)
        self.assertEqual(range_check(str(5)), 5, "Should be 5")


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load in testing parameters
        with open(join(FIX_DIR, "test_pars.json"), "r", encoding="utf-8") as j_id:
            pars = json.load(j_id)

        # Extract parameters in format necessary for x_epi
        cls.seqs = []
        for idx, par in enumerate(pars.values()):
            seq_out = join(FIX_DIR, "test_" + par["name"])

            # Add general metabolites
            args = f"-out {seq_out}"
            args += create_arg_str(par["general"])

            # Add spectra if necessary
            if par["spec"] is True:
                args += " -run_spec BOTH"

            # Add parameters for reach metabolite
            for met in par["mets"]:
                if len(met) > 0:
                    args += " -met" + create_arg_str(met)

            # Run sequence
            main(argv=args.split())
            cls.seqs.append(seq_out + ".seq")

    def test_main_help(self):
        with self.assertRaises(SystemExit):
            main(argv=["-h"])

    def test_main_help_no_met(self):
        with self.assertRaises(SystemExit):
            main(argv=["-size", "16", "16", "16"])

    def test_main_seq(self):
        # Compare each sequence to reference
        for seq in self.seqs:
            cmp = filecmp.cmp(seq, seq.replace("test_", ""), shallow=False)
            self.assertTrue(cmp, msg=f"{seq} failed")

    def test_main_json(self):
        for seq in self.seqs:
            test_path = seq.replace(".seq", ".json")
            ref_path = test_path.replace("test_", "")
            comp = comp_json(test_path, ref_path)
            self.assertEqual(comp, {}, "Should return an empty dictionary")

    @classmethod
    def tearDownClass(cls):
        for seq in cls.seqs:
            remove(seq)
            remove(seq.replace(".seq", ".json"))
            remove(seq.replace(".seq", "_kspace.npy"))


if __name__ == "__main__":
    unittest.main(buffer=True)
