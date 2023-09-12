"""
Very simple test for ui module created by QtDesigner
"""

import unittest
from PyQt5.QtWidgets import QApplication, QMainWindow, QDoubleSpinBox
from x_epi.ui import Ui_MainWindow


class TestSetupUi(unittest.TestCase):

    def test_setupUi_create(self):

        #Setup ui window object
        app = QApplication([])
        win = QMainWindow()
        ui_win = Ui_MainWindow()
        ui_win.setupUi(win)

        #Basic check to see if ui object is load correctly
        self.assertIs(type(ui_win.dbl_spin_b0), QDoubleSpinBox)

if __name__ == '__main__':
    unittest.main()
