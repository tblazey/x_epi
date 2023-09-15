"""
Testing for x_epi class
"""

import filecmp
import json
from os.path import abspath, basename, dirname, join, splitext
from os import remove
import unittest
from jsoncomparison import Compare
from x_epi.seq import XSeq
from x_epi.utils import BASE_DIR

FIX_DIR = abspath(join(BASE_DIR, "..", "tests/fixtures"))


class TestXSeq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define sequence parameters for testing
        par_1 = {
            "general": {
                "grad_spoil": True,
                "n_rep": 2,
                "alt_read": True,
                "alt_pha": True,
                "alt_slc": True,
                "n_echo": 3,
                "tv": 1000,
                "ts": 2500,
            },
            "mets": [{"pf_pe2": 0.75, "z_centric": True}, {"use_sinc": True}],
            "create": {"return_plot": True},
            "spec": False,
        }
        par_2 = {
            "general": {
                "ramp_samp": True,
                "max_slew": 120,
                "max_grad": 60,
                "acq_3d": False,
                "symm_ro": False,
                "grad_spoil": True,
                "n_echo": 3,
                "tr": 500,
                "delta_te": 150,
            },
            "mets": [
                {"ro_os": 2.0, "pf_pe": 0.75, "use_sinc": True},
                {"use_sinc": False, "freq_off": 50},
            ],
            "create": {},
            "spec": True,
        }
        par_3 = {
            "general": {"no_pe": True, "no_slc": True, "n_reps": 2},
            "mets": [{}],
            "create": {"no_reps": True},
            "spec": False,
        }
        par_4 = {
            "general": {"no_pe": True, "acq_3d": True, "symm_ro": False},
            "mets": [{}],
            "create": {},
            "spec": False,
        }
        pars = [par_1, par_2, par_3, par_4]

        # Loop through parameters
        cls.seqs = []
        for idx, par in enumerate(pars):
            # Add sequence to class
            seq = XSeq(**par["general"])

            # Add metabolites
            for met in par["mets"]:
                seq.add_met(**met)

            # Add spectra if necessary
            if par["spec"] is True:
                seq.add_spec(run_spec="BOTH")

            # Write out names
            seq.out_name = f"seq_{idx + 1}.seq"
            seq.out_path = join(FIX_DIR, "test_" + seq.out_name)

            # Create the sequence
            seq.create_seq(**par["create"])
            cls.seqs.append(seq)

            # Write output
            seq.write(seq.out_path)
            seq.save_params(splitext(seq.out_path)[0])

    def test_seqs(self):
        # Compare each sequence to reference
        for seq in self.seqs:
            cmp = filecmp.cmp(seq.out_path, join(FIX_DIR, seq.out_name), shallow=False)
            self.assertTrue(cmp, msg=f"{seq.out_name} failed")

    def test_save_params(self):
        for seq in self.seqs:
            # Load in json dictionaries
            test_path = splitext(seq.out_path)[0] + ".json"
            with open(test_path, "r", encoding="utf-8") as j_id:
                test_dic = json.load(j_id)
            ref_path = join(FIX_DIR, splitext(seq.out_name)[0] + ".json")
            with open(ref_path, "r", encoding="utf-8") as j_id:
                ref_dic = json.load(j_id)

            # Edit rf paths because they won't be the same if this is run on another system
            for ref_met, test_met in zip(ref_dic["mets"], test_dic["mets"]):
                ref_met["grd_path"] = join(
                    dirname(test_met["grd_path"]), basename(ref_met["grd_path"])
                )
                ref_met["rf_path"] = join(
                    dirname(test_met["rf_path"]), basename(ref_met["rf_path"])
                )

            # Run comparison between dictionaries
            cmp = Compare().check(test_dic, ref_dic)
            self.assertEqual(cmp, {}, "Should return an empty dictionary")

    @classmethod
    def tearDownClass(cls):
        for seq in cls.seqs:
            remove(seq.out_path)
            remove(splitext(seq.out_path)[0] + ".json")


if __name__ == "__main__":
    unittest.main()
