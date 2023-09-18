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
            cmp = filecmp.cmp(seq.out_path, join(FIX_DIR, seq.out_name), shallow=False)
            self.assertTrue(cmp, msg=f"{seq.out_name} failed")

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
