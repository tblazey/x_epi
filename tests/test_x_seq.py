"""
Testing for x_epi class
"""

import json
from os.path import abspath, basename, dirname, join, splitext
from os import remove
import unittest
from jsoncomparison import Compare
import numpy as np
import pypulseq as pp
from x_epi.seq import XSeq
from x_epi.utils import BASE_DIR

FIX_DIR = abspath(join(BASE_DIR, "..", "tests/fixtures"))


def comp_json(test_path, ref_path):
    # Load in json dictionaries
    with open(test_path, "r", encoding="utf-8") as j_id:
        test_dic = json.load(j_id)
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
    return Compare().check(test_dic, ref_dic)


class TestXSeq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load in testing parameters
        with open(join(FIX_DIR, "test_pars.json"), "r", encoding="utf-8") as j_id:
            pars = json.load(j_id)

        # Loop through parameters
        cls.seqs = []
        for idx, par in enumerate(pars.values()):
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
        
            # Read in reference sequence
            ref_seq = pp.Sequence()
            ref_seq.read(join(FIX_DIR, seq.out_name))
            
            # Get gradient/rf values for each sequence
            seq_arrs = seq.waveforms_and_times(append_RF=True)
            ref_arrs = ref_seq.waveforms_and_times(append_RF=True)
            
            # Check that the waveforms arrays are the same
            check = True
            for s_arr, r_arr in zip(seq_arrs[0] + list(seq_arrs[1:]), ref_arrs[0] + list(ref_arrs[1:])):
                if s_arr.shape != r_arr.shape:
                    check = False
                elif np.allclose(s_arr, r_arr, atol=1E-3) is False:
                    check = False

            self.assertTrue(check, msg=f"{seq.out_name} failed")

    def test_save_params(self):
        for seq in self.seqs:
            test_path = splitext(seq.out_path)[0] + ".json"
            ref_path = join(FIX_DIR, splitext(seq.out_name)[0] + ".json")
            comp = comp_json(test_path, ref_path)
            self.assertEqual(comp, {}, "Should return an empty dictionary")

    @classmethod
    def tearDownClass(cls):
        for seq in cls.seqs:
            remove(seq.out_path)
            remove(splitext(seq.out_path)[0] + ".json")


if __name__ == "__main__":
    unittest.main()
