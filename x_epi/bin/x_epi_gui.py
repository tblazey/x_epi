"""
PyQT gui script for creating  EPI sequences with x_epi
"""

# Load libraries
from copy import deepcopy
import glob
import json
from itertools import product
import os
import subprocess as sp
import sys

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QButtonGroup,
)
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
import x_epi


class ScalarFormatterForceFormat(ScalarFormatter):
    """
    Custom formatter for matplotlib. Disables sci. notation and
    displays a float with one decimal place
    """

    # https://tinyurl.com/2s3fmjsy
    def __init__(self):
        super().__init__()
        self.set_powerlimits((0, 0))

    def _set_format(self):
        self.format = "%+1.1f"


basedir = os.path.dirname(__file__)


class MyMainWindow(QMainWindow, x_epi.ui.Ui_MainWindow):
    """
    Window class for PyQ5 app. Inherits from QtDesigner output
    """

    def __init__(self, json_path=None):
        """
        Creates x_epi_app window
        """

        super().__init__()

        # Add UI to class
        self.setupUi(self)
        self.Tabs.setCurrentIndex(0)

        # Acquisition mode button group
        self.button_group_acq_3d = QButtonGroup()
        self.button_group_acq_3d.addButton(self.radio_acq_2d)
        self.button_group_acq_3d.addButton(self.radio_acq_3d)

        # Readout button group
        self.button_group_symm_ro = QButtonGroup()
        self.button_group_symm_ro.addButton(self.radio_fly_ro)
        self.button_group_symm_ro.addButton(self.radio_symm_ro)

        # Phase button group
        self.button_group_no_pe = QButtonGroup()
        self.button_group_no_pe.addButton(self.radio_phase_on)
        self.button_group_no_pe.addButton(self.radio_phase_off)

        # Slice button group
        self.button_group_no_slc = QButtonGroup()
        self.button_group_no_slc.addButton(self.radio_slice_grad_on)
        self.button_group_no_slc.addButton(self.radio_slice_grad_off)

        # Spoil button group
        self.button_group_grad_spoil = QButtonGroup()
        self.button_group_grad_spoil.addButton(self.radio_spoil_off)
        self.button_group_grad_spoil.addButton(self.radio_spoil_on)

        # Ramp sampling button group
        self.button_group_ramp_samp = QButtonGroup()
        self.button_group_ramp_samp.addButton(self.radio_ramp_off)
        self.button_group_ramp_samp.addButton(self.radio_ramp_on)

        # Prep for figure
        self.figure = plt.figure(figsize=(10, 5.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot_layout = QVBoxLayout()
        self.plot_layout.addWidget(self.canvas, alignment=QtCore.Qt.AlignTop)
        self.central_widget.setLayout(self.plot_layout)

        # Load in sequence information
        self.met_idx = 0
        self.updating = False
        self.load_json(json_path)

        # Create initial sequence
        self.update_for_plot()

        # Update ui
        self.dic_to_ui()
        self.toggle_ro_os()
        self.update_combo_met()
        self.update_combo_pe_dir()
        self.update_combo_plot()

        #####################
        ###General Updates###
        #####################

        # FOV
        self.map_signal(
            self.spin_fov_ro.editingFinished, [self.ui_to_dic, self.update_localizer]
        )
        self.map_signal(
            self.spin_fov_pe.editingFinished, [self.ui_to_dic, self.update_localizer]
        )
        self.map_signal(
            self.spin_fov_slc.editingFinished, [self.ui_to_dic, self.update_localizer]
        )

        # Offsets
        self.map_signal(
            self.dbl_spin_ro_off.editingFinished,
            [self.ui_to_dic, self.update_localizer],
        )
        self.map_signal(
            self.dbl_spin_pe_off.editingFinished,
            [self.ui_to_dic, self.update_localizer],
        )
        self.map_signal(
            self.dbl_spin_slc_off.editingFinished,
            [self.ui_to_dic, self.update_localizer],
        )

        # Number of metabolites, reps, averages
        self.map_signal(
            self.spin_n_met.textChanged, [self.update_combo_met, self.ui_to_dic]
        )
        self.map_signal(
            self.spin_n_rep.textChanged, [self.ui_to_dic, self.update_flips]
        )
        self.map_signal(
            self.spin_n_avg.textChanged, [self.ui_to_dic, self.update_flips]
        )

        # Bandwidth and slice axis
        self.spin_rbw.textChanged.connect(self.ui_to_dic)
        self.combo_slice_axis.currentIndexChanged.connect(self.ui_to_dic)

        # Polarity options
        self.check_alt_read.stateChanged.connect(self.ui_to_dic)
        self.check_alt_pha.stateChanged.connect(self.ui_to_dic)
        self.check_alt_slc.stateChanged.connect(self.ui_to_dic)

        # Timing options
        self.dbl_spin_tr.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_tv.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_ts.valueChanged.connect(self.ui_to_dic)

        # Orientation options
        self.map_signal(
            self.combo_ori.currentIndexChanged,
            [self.update_combo_pe_dir, self.ui_to_dic, self.update_localizer],
        )
        self.map_signal(
            self.combo_pe_dir.currentIndexChanged,
            [self.ui_to_dic, self.update_localizer],
        )

        # Echo options
        self.spin_n_echo.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_delta_te.valueChanged.connect(self.ui_to_dic)

        # Spectra options
        self.spin_spec_size.textChanged.connect(self.ui_to_dic)
        self.spin_spec_bw.textChanged.connect(self.ui_to_dic)
        self.dbl_spin_spec_flip.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_spec_tr.valueChanged.connect(self.ui_to_dic)
        self.spin_spec_n.textChanged.connect(self.ui_to_dic)

        # Acquisition options
        self.button_group_acq_3d.buttonToggled.connect(self.ui_to_dic)
        self.button_group_symm_ro.buttonToggled.connect(self.ui_to_dic)
        self.button_group_no_pe.buttonToggled.connect(self.ui_to_dic)
        self.button_group_no_slc.buttonToggled.connect(self.ui_to_dic)
        self.button_group_grad_spoil.buttonToggled.connect(self.ui_to_dic)
        self.map_signal(
            self.button_group_spec.buttonToggled, [self.spec_enable, self.ui_to_dic]
        )
        self.map_signal(
            self.button_group_ramp_samp.buttonToggled,
            [self.toggle_ro_os, self.ui_to_dic],
        )
        self.dbl_spin_ro_os.valueChanged.connect(self.ui_to_dic)

        ########################
        ###Metabolite updates###
        ########################

        # If pulse type is changed
        self.map_signal(
            self.combo_use_sinc.currentIndexChanged,
            [self.update_rf_type, self.ui_to_dic],
        )

        # If load rf/gradient buttons are pressed.
        self.map_signal(
            self.button_load_rf.clicked, [self.update_ssrf_rf, self.ui_to_dic]
        )
        self.map_signal(
            self.button_load_grd.clicked, [self.update_ssrf_grd, self.ui_to_dic]
        )

        # If formula is updated
        self.line_formula.textChanged.connect(self.ui_to_dic)

        # If sinc options are changed
        self.dbl_spin_sinc_dur.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_sinc_frac.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_sinc_tbw.valueChanged.connect(self.ui_to_dic)

        # If excitation options are changed
        self.dbl_spin_freq_off.valueChanged.connect(self.ui_to_dic)
        self.map_signal(
            self.dbl_spin_flip.valueChanged, [self.ui_to_dic, self.update_flips]
        )
        self.line_name.textChanged.connect(self.ui_to_dic)

        # Acquisition options
        self.spin_size_ro.textChanged.connect(self.ui_to_dic)
        self.spin_size_pe.textChanged.connect(self.ui_to_dic)
        self.spin_size_slc.textChanged.connect(self.ui_to_dic)
        self.dbl_spin_pf_pe.valueChanged.connect(self.ui_to_dic)
        self.map_signal(
            self.dbl_spin_pf_pe2.valueChanged, ([self.ui_to_dic, self.update_flips])
        )
        self.check_z_centric.stateChanged.connect(self.ui_to_dic)

        ####################
        ###System Updates###
        ####################

        # Limits
        self.dbl_spin_max_grad.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_max_slew.valueChanged.connect(self.ui_to_dic)

        # Misc
        self.dbl_spin_b0.valueChanged.connect(self.ui_to_dic)
        self.combo_nuc.currentIndexChanged.connect(self.ui_to_dic)

        # RF timing
        self.dbl_spin_rf_ringdown_time.valueChanged.connect(self.ui_to_dic)
        self.dbl_spin_rf_dead_time.valueChanged.connect(self.ui_to_dic)

        ###################
        ###Menu Updates###
        ###################

        # Menu bar actions
        self.menu_open.triggered.connect(self.load_pars)
        self.menu_save.triggered.connect(self.save_seq)
        self.map_signal(
            self.menu_load_localizer.triggered,
            [self.load_localizer, self.update_combo_plot, self.update_localizer],
        )

        # If metabolite selection is changed on ui bar
        self.map_signal(
            self.combo_met.currentIndexChanged, [self.update_met_idx, self.dic_to_ui]
        )

        # If plot type is changed
        self.combo_plot.currentIndexChanged.connect(self.plot)

        # If time update/save buttons are pressed
        self.button_update.clicked.connect(self.update_for_plot)
        self.button_save.clicked.connect(self.save_seq)

    def map_signal(self, signal, funcs):
        """
        Function to map a signal to multiple functions

        Parameters
        ----------
        signal : object
           Pyqt signal
        funcs : list
           List of functions to amp
        """
        list(map(signal.connect, funcs))

    def load_json(self, json_path, use_default=True):
        """
        Load in json parameter file defining XSeq sequence and save it into param_dic

        Parameters
        ----------
        json_path : str
           Path to json file
        use_default : bool
           Load in default parameters if json_path cannot be read
        """

        try:
            with open(json_path, "r", encoding="utf-8") as j_id:
                self.param_dic = json.load(j_id)
        except TypeError:
            if use_default is True:
                j_path = os.path.join(x_epi.RES_DIR, "default.json")
                with open(j_path, "r", encoding="utf-8") as j_id:
                    self.param_dic = json.load(j_id)

    def toggle_ro_os(self):
        """
        Shows/hides readout oversampling ui elements
        """
        self.updating = True
        if self.radio_ramp_off.isChecked():
            status = False
        else:
            status = True
        self.dbl_spin_ro_os.setVisible(status)
        self.label_ro_os.setVisible(status)
        self.updating = False

    def dic_to_seq(self, return_plot=True, no_reps=False):
        """
        Converts parameter dictionary to XSeq sequence

        Parameters
        ----------
        return_plot : bool
           Returns a sequence with no reps or averages for plotting
        no_reps : bool
           Ignores averages and reps when computing sequence
        """

        # Define common sequence parameters
        self.seq = x_epi.XSeq(**self.param_dic)

        # Add spectra options if needed
        try:
            if self.param_dic["run_spec"] != "NO":
                self.seq.add_spec(**self.param_dic)
        except KeyError:
            pass

        # Add metabolite options.
        for i in range(self.param_dic["n_met"]):
            self.seq.add_met(**self.param_dic["mets"][i])

        # Create sequence
        self.plot_seq = self.seq.create_seq(return_plot=return_plot, no_reps=no_reps)

    def ui_to_dic(self):
        """
        Updates dictionary based on UI
        """
        if self.updating is True:
            return

        # General options
        self.param_dic["fov"][0] = self.spin_fov_ro.value()
        self.param_dic["fov"][1] = self.spin_fov_pe.value()
        self.param_dic["fov"][2] = self.spin_fov_slc.value()
        self.param_dic["rbw"] = self.spin_rbw.value()
        self.param_dic["tr"] = self.dbl_spin_tr.value()
        self.param_dic["tv"] = self.dbl_spin_tv.value()
        self.param_dic["ts"] = self.dbl_spin_ts.value()
        self.param_dic["ro_off"] = self.dbl_spin_ro_off.value()
        self.param_dic["pe_off"] = self.dbl_spin_pe_off.value()
        self.param_dic["slc_off"] = self.dbl_spin_slc_off.value()
        self.param_dic["n_echo"] = self.spin_n_echo.value()
        self.param_dic["delta_te"] = self.dbl_spin_delta_te.value()
        self.param_dic["n_met"] = self.spin_n_met.value()
        self.param_dic["n_avg"] = self.spin_n_avg.value()
        self.param_dic["n_rep"] = self.spin_n_rep.value()
        self.param_dic["symm_ro"] = self.button_group_symm_ro.buttons()[1].isChecked()
        self.param_dic["acq_3d"] = self.button_group_acq_3d.buttons()[1].isChecked()
        self.param_dic["no_pe"] = self.button_group_no_pe.buttons()[1].isChecked()
        self.param_dic["no_slc"] = self.button_group_no_slc.buttons()[1].isChecked()
        self.param_dic["grad_spoil"] = self.button_group_grad_spoil.buttons()[
            1
        ].isChecked()
        self.param_dic["ramp_samp"] = self.button_group_ramp_samp.buttons()[
            1
        ].isChecked()
        self.param_dic["slice_axis"] = self.combo_slice_axis.currentText()
        self.param_dic["alt_read"] = self.check_alt_read.isChecked()
        self.param_dic["alt_pha"] = self.check_alt_pha.isChecked()
        self.param_dic["alt_slc"] = self.check_alt_slc.isChecked()
        self.param_dic["ro_os"] = self.dbl_spin_ro_os.value()
        self.param_dic["b0"] = self.dbl_spin_b0.value()
        self.param_dic["nuc"] = self.combo_nuc.currentText()
        self.param_dic["max_grad"] = self.dbl_spin_max_grad.value()
        self.param_dic["max_slew"] = self.dbl_spin_max_slew.value()
        self.param_dic["rf_ringdown_time"] = self.dbl_spin_rf_ringdown_time.value()
        self.param_dic["rf_dead_time"] = self.dbl_spin_rf_dead_time.value()
        self.param_dic["ori"] = self.combo_ori.currentText()
        self.param_dic["pe_dir"] = self.combo_pe_dir.currentText()

        # Spectra options
        if self.check_spec_start.isChecked() and self.check_spec_end.isChecked():
            self.param_dic["run_spec"] = "BOTH"
        elif self.check_spec_start.isChecked():
            self.param_dic["run_spec"] = "START"
        elif self.check_spec_end.isChecked():
            self.param_dic["run_spec"] = "END"
        else:
            self.param_dic["run_spec"] = "NO"
        self.param_dic["spec_size"] = self.spin_spec_size.value()
        self.param_dic["spec_bw"] = self.spin_spec_bw.value()
        self.param_dic["spec_flip"] = self.dbl_spin_spec_flip.value()
        self.param_dic["spec_tr"] = self.dbl_spin_spec_tr.value()
        self.param_dic["spec_n"] = self.spin_spec_n.value()

        # Save info from current metabolite
        curr_met = self.param_dic["mets"][self.met_idx]
        curr_met["formula"] = self.line_formula.text()
        curr_met["use_sinc"] = self.combo_use_sinc.currentText() == "Sinc"
        curr_met["flip"] = self.dbl_spin_flip.value()
        curr_met["freq_off"] = self.dbl_spin_freq_off.value()
        curr_met["sinc_dur"] = self.dbl_spin_sinc_dur.value()
        curr_met["sinc_frac"] = self.dbl_spin_sinc_frac.value()
        curr_met["sinc_tbw"] = self.dbl_spin_sinc_tbw.value()
        curr_met["size"][0] = self.spin_size_ro.value()
        curr_met["size"][1] = self.spin_size_pe.value()
        curr_met["size"][2] = self.spin_size_slc.value()
        curr_met["pf_pe"] = self.dbl_spin_pf_pe.value()
        curr_met["pf_pe2"] = self.dbl_spin_pf_pe2.value()
        curr_met["z_centric"] = self.check_z_centric.isChecked()
        curr_met["grd_path"] = self.line_grd_path.text()
        curr_met["rf_path"] = self.line_rf_path.text()
        curr_met["b1_max"] = self.line_b1_max.text()
        curr_met["grd_max"] = self.line_grd_max.text()
        curr_met["grd_delta"] = self.line_grd_delta.text()
        curr_met["rf_delta"] = self.line_rf_delta.text()
        curr_met["name"] = self.line_name.text()

    def dic_to_ui(self):
        """
        Updates UI based on dictionary"
        """
        if self.updating is True:
            return

        # Set parameter values that are the same for all metabolites
        self.updating = True
        self.spin_fov_ro.setValue(self.param_dic["fov"][0])
        self.spin_fov_pe.setValue(self.param_dic["fov"][1])
        self.spin_fov_slc.setValue(self.param_dic["fov"][2])
        self.spin_rbw.setValue(int(self.param_dic["rbw"]))
        self.dbl_spin_tr.setValue(self.param_dic["tr"])
        self.dbl_spin_tv.setValue(self.param_dic["tv"])
        self.dbl_spin_ts.setValue(self.param_dic["ts"])
        self.dbl_spin_ro_off.setValue(self.param_dic["ro_off"])
        self.dbl_spin_pe_off.setValue(self.param_dic["pe_off"])
        self.dbl_spin_slc_off.setValue(self.param_dic["slc_off"])
        self.spin_n_echo.setValue(self.param_dic["n_echo"])
        self.dbl_spin_delta_te.setValue(self.param_dic["delta_te"])
        self.spin_n_met.setValue(self.param_dic["n_met"])
        self.spin_n_avg.setValue(self.param_dic["n_avg"])
        self.spin_n_rep.setValue(self.param_dic["n_rep"])
        self.button_group_symm_ro.buttons()[self.param_dic["symm_ro"]].setChecked(True)
        self.button_group_acq_3d.buttons()[self.param_dic["acq_3d"]].setChecked(True)
        self.button_group_no_pe.buttons()[self.param_dic["no_pe"]].setChecked(True)
        self.button_group_no_slc.buttons()[self.param_dic["no_slc"]].setChecked(True)
        self.button_group_grad_spoil.buttons()[self.param_dic["grad_spoil"]].setChecked(
            True
        )
        self.button_group_ramp_samp.buttons()[self.param_dic["ramp_samp"]].setChecked(
            True
        )
        self.combo_slice_axis.setCurrentText(self.param_dic["slice_axis"])
        self.check_alt_read.setChecked(self.param_dic["alt_read"])
        self.check_alt_pha.setChecked(self.param_dic["alt_pha"])
        self.check_alt_slc.setChecked(self.param_dic["alt_slc"])
        self.dbl_spin_ro_os.setValue(self.param_dic["ro_os"])
        self.dbl_spin_b0.setValue(self.param_dic["b0"])
        self.combo_nuc.setCurrentText(self.param_dic["nuc"])
        self.dbl_spin_max_grad.setValue(self.param_dic["max_grad"])
        self.dbl_spin_max_slew.setValue(self.param_dic["max_slew"])
        self.dbl_spin_rf_ringdown_time.setValue(self.param_dic["rf_ringdown_time"])
        self.dbl_spin_rf_dead_time.setValue(self.param_dic["rf_dead_time"])
        self.combo_ori.setCurrentText(self.param_dic["ori"])
        self.combo_pe_dir.setCurrentText(self.param_dic["pe_dir"])

        # Try to set spectra info
        try:
            if (
                self.param_dic["run_spec"] == "START"
                or self.param_dic["run_spec"] == "BOTH"
            ):
                self.check_spec_start.setChecked(True)
            if (
                self.param_dic["run_spec"] == "END"
                or self.param_dic["run_spec"] == "BOTH"
            ):
                self.check_spec_end.setChecked(True)
            self.spin_spec_size.setValue(self.param_dic["spec_size"])
            self.spin_spec_bw.setValue(self.param_dic["spec_bw"])
            self.dbl_spin_spec_flip.setValue(self.param_dic["spec_flip"])
            self.dbl_spin_spec_tr.setValue(self.param_dic["spec_tr"])
            self.spin_spec_n.setValue(self.param_dic["spec_n"])
        except KeyError:
            pass

        # Update metabolite info
        self.update_met_ui()
        self.updating = False

    def spec_enable(self):
        """
        Function to enable spectra options based on UI
        """

        if self.check_spec_start.isChecked() or self.check_spec_end.isChecked():
            state = True
        else:
            state = False
        self.spin_spec_size.setEnabled(state)
        self.spin_spec_bw.setEnabled(state)
        self.dbl_spin_spec_flip.setEnabled(state)
        self.spin_n_spec.setEnabled(state)
        self.dbl_spin_spec_tr.setEnabled(state)

    def update_rf_type(self):
        """
        Quick function to change RF pane
        """
        if self.combo_use_sinc.currentIndex() == 0:
            self.stacked_pulse_type.setCurrentIndex(0)
        else:
            self.stacked_pulse_type.setCurrentIndex(1)

    def update_localizer(self):
        """
        Function to update localizer
        """

        if self.combo_plot.currentText() == "Localizer":
            self.plot()

    def plot(self):
        """
        Function to add various plots to top window
        """

        plot_type = self.combo_plot.currentText()
        if plot_type is None or plot_type == "":
            return
        self.figure.clear()

        # Make sure we don't have to update plot
        met = self.met_idx
        if plot_type != "Localizer" and len(self.waves[2]) <= met:
            self.update_for_plot()

        # Determine which plot to do
        if plot_type in ("2D k-space", "3D k-space"):
            if plot_type == "2D k-space":
                ax = self.figure.add_subplot()
                ax.grid()
                ax.plot(self.waves[2][met][0, :], self.waves[2][met][1, :])
                ax.scatter(
                    self.waves[3][met][0, :], self.waves[3][met][1, :], c="red", s=10
                )
                ax.axis("equal")
            else:
                ax = self.figure.add_subplot(projection="3d")
                ax.set_zlabel(r"$k_z$ ($mm^{-1}$)")
                ax.plot(
                    self.waves[0][met][0, :],
                    self.waves[0][met][1, :],
                    self.waves[0][met][2, :],
                )
                ax.scatter(
                    self.waves[1][met][0, :],
                    self.waves[1][met][1, :],
                    self.waves[1][met][2, :],
                    c="red",
                    s=2,
                )

                # Get "coordinates" for each dim
                k_extent = np.max(np.abs(self.waves[0][met]), axis=1)
                k_coords = []
                for i in range(3):
                    k_coords.append([-k_extent[i], 0, k_extent[i]])

                # Loop through dimensions again
                for i in range(3):
                    # Make grids
                    dims = [0, 1, 2]
                    dims.remove(i)
                    c1, c2 = np.meshgrid(
                        k_coords[dims[0]], k_coords[dims[1]], indexing="ij"
                    )
                    coords = [[], [], []]
                    coords[dims[0]] = c1
                    coords[dims[1]] = c2
                    coords[i] = c1 * 0

                    # Add plane
                    ax.plot_surface(
                        coords[0], coords[1], coords[2], color="gray", alpha=0.5
                    )

            # Common k-space plot optionsn
            ax.set_xlabel(r"$k_x$ ($mm^{-1}$)")
            ax.set_ylabel(r"$k_y$ ($mm^{-1}$)")

        elif plot_type == "Waveforms":
            # Common options
            t = (
                np.arange(self.waves[4][self.met_idx].shape[1])
                * self.plot_seq.system.grad_raster_time
                * 1e3
            )

            # Magnitude plot
            ax_mag = self.figure.add_subplot(5, 1, 1)
            ax_mag.grid()
            ax_mag.plot(t, np.abs(self.waves[5][self.met_idx]), linewidth=0.5)
            ax_mag.set_ylabel("Mag.\n(Hz)", fontweight="bold", rotation=0)
            ax_mag.xaxis.set_ticklabels([])
            ax_mag.yaxis.set_label_coords(-0.1, 0.3)
            ax_mag.yaxis.set_major_formatter(ScalarFormatterForceFormat())

            # Phase plot
            ax_phase = self.figure.add_subplot(5, 1, 2)
            ax_phase.grid()
            ax_phase.plot(t, np.angle(self.waves[5][self.met_idx]), linewidth=0.5)
            ax_phase.set_ylabel("Phase\n(rad)", fontweight="bold", rotation=0)
            ax_phase.xaxis.set_ticklabels([])
            ax_phase.yaxis.set_label_coords(-0.1, 0.3)
            ax_phase.yaxis.set_major_formatter(ScalarFormatterForceFormat())

            # X gradient plot
            ax_x = self.figure.add_subplot(5, 1, 3)
            ax_x.grid()
            ax_x.plot(t, self.waves[4][self.met_idx][0, :], linewidth=0.5)
            ax_x.set_ylabel("Gx\n(Hz/m)", fontweight="bold", rotation=0)
            ax_x.xaxis.set_ticklabels([])
            ax_x.yaxis.set_label_coords(-0.1, 0.3)
            ax_x.yaxis.set_major_formatter(ScalarFormatterForceFormat())

            # Y gradient plot
            ax_y = self.figure.add_subplot(5, 1, 4)
            ax_y.grid()
            ax_y.plot(t, self.waves[4][self.met_idx][1, :], linewidth=0.5)
            ax_y.set_ylabel("Gy\n(Hz/m)", fontweight="bold", rotation=0)
            ax_y.xaxis.set_ticklabels([])
            ax_y.yaxis.set_label_coords(-0.1, 0.3)
            ax_y.yaxis.set_major_formatter(ScalarFormatterForceFormat())

            # Z gradient plot
            ax_z = self.figure.add_subplot(5, 1, 5)
            ax_z.grid()
            ax_z.plot(t, self.waves[4][self.met_idx][2, :], linewidth=0.5)
            ax_z.set_xlabel("Time (ms)", fontweight="bold")
            ax_z.set_ylabel("Gz\n(Hz/m)", fontweight="bold", rotation=0)
            ax_z.yaxis.set_label_coords(-0.1, 0.3)
            ax_z.yaxis.set_major_formatter(ScalarFormatterForceFormat())

            # Adjust spacing
            self.figure.subplots_adjust(hspace=0.1)

        else:
            # Epi stuff
            epi_fov = self.param_dic["fov"]

            # Get coordinates of bounding box
            verts = np.zeros((3, 8))
            for idx, coords in enumerate(
                product(
                    [-epi_fov[0] / 2, epi_fov[0] / 2],
                    [-epi_fov[1] / 2, epi_fov[1] / 2],
                    [-epi_fov[2] / 2, epi_fov[2] / 2],
                )
            ):
                verts[
                    :,
                    idx,
                ] = coords
            offsets = np.array(
                [
                    self.param_dic["ro_off"],
                    self.param_dic["pe_off"],
                    self.param_dic["slc_off"],
                ]
            )

            # Define what orientation we are going to use
            use_ori = self.param_dic["ori"].lower()[0:3]
            pha_sec = self.param_dic["pe_dir"] != self.loc_dic[use_ori]["pe_prim"]
            if pha_sec is True:
                pha_swap = np.array(
                    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
            else:
                pha_swap = np.eye(4)
            use_aff = self.ori_dic[use_ori]["aff"] @ pha_swap
            use_codes = nib.aff2axcodes(use_aff)
            use_codes_neg = nib.aff2axcodes(
                use_aff
                @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            )
            verts = use_aff[0:3, 0:3] @ (verts + offsets[:, np.newaxis])

            # Define colors for readout, phase encoding, and slice directions
            colors = ["#006bb6", "#00b64b", "#b6006b"]

            # Add localizer to plot
            ax = self.figure.subplots(1, 3)
            for axis, ori in zip(ax, ["sag", "cor", "tra"]):
                # Add image for current orientation
                axis.matshow(
                    self.loc_dic[ori]["img"][:, :, self.loc_dic[ori]["slice"]].T,
                    cmap="gray",
                    origin="lower",
                )
                axis.set_title(
                    self.ori_dic[ori]["name"], fontweight="bold", fontsize=16
                )
                axis.axis("off")

                # Define color for labels
                if self.loc_dic[ori]["codes+"][0] in use_codes:
                    x_color = colors[use_codes.index(self.loc_dic[ori]["codes+"][0])]
                elif self.loc_dic[ori]["codes+"][0] in use_codes_neg:
                    x_color = colors[
                        use_codes_neg.index(self.loc_dic[ori]["codes+"][0])
                    ]
                if self.loc_dic[ori]["codes+"][1] in use_codes:
                    y_color = colors[use_codes.index(self.loc_dic[ori]["codes+"][1])]
                elif self.loc_dic[ori]["codes+"][1] in use_codes_neg:
                    y_color = colors[
                        use_codes_neg.index(self.loc_dic[ori]["codes+"][1])
                    ]

                # Add axis label codes
                axis.annotate(
                    self.loc_dic[ori]["codes+"][1],
                    xy=[0.5, 0.9],
                    xycoords="axes fraction",
                    fontweight="bold",
                    fontsize=14,
                    rotation=0,
                    color=y_color,
                )
                axis.annotate(
                    self.loc_dic[ori]["codes+"][0],
                    xy=[0.9, 0.5],
                    xycoords="axes fraction",
                    fontweight="bold",
                    fontsize=14,
                    rotation=0,
                    color=x_color,
                )
                axis.annotate(
                    self.loc_dic[ori]["codes-"][1],
                    xy=[0.5, 0.05],
                    xycoords="axes fraction",
                    fontweight="bold",
                    fontsize=14,
                    rotation=0,
                    color=y_color,
                )
                axis.annotate(
                    self.loc_dic[ori]["codes-"][0],
                    xy=[0.05, 0.5],
                    xycoords="axes fraction",
                    fontweight="bold",
                    fontsize=14,
                    rotation=0,
                    color=x_color,
                )

                # Get voxel coordinates of box
                vox_c = (
                    self.loc_dic[ori]["scan_to_vox"][0:3, 0:3] @ verts
                    + self.loc_dic[ori]["scan_to_vox"][0:3, 3][:, np.newaxis]
                )
                box_min = np.min(vox_c[0:2, :], axis=1)
                box_delta = np.max(vox_c[0:2, :], axis=1) - box_min

                # Make box
                rect = Rectangle(
                    box_min, box_delta[0], box_delta[1], color="yellow", fill=None
                )
                axis.add_patch(rect)

        plt.tight_layout()
        self.canvas.draw()

    def update_ssrf_grd(self):
        """
        Load in new ssrf gradient data
        """
        self.updating = True

        try:
            # Load in gradient data
            grd_path = QFileDialog.getOpenFileName(self, "Load Gradient File")[0]
            _, grd_max, grd_delta = x_epi.load_ssrf_grad(grd_path)

            # Update interface
            self.line_grd_path.setText(str(grd_path))
            self.line_grd_max.setText(str(grd_max))
            self.line_grd_delta.setText(str(grd_delta * 1e6))

        except TypeError:
            self.update_ssrf_grd()

        self.updating = False

    def update_ssrf_rf(self):
        """
        Load in new ssrf rf data
        """

        self.updating = True

        try:
            # Load in rf data
            rf_path = QFileDialog.getOpenFileName(self, "Load RF File")[0]
            _, _, b1_max, rf_delta = x_epi.load_ssrf_rf(rf_path)

            # Update interface
            self.line_rf_path.setText(str(rf_path))
            self.line_b1_max.setText(str(b1_max))
            self.line_rf_delta.setText(str(rf_delta))

        except TypeError:
            self.update_ssrf_rf()

        self.updating = False

    # Change metabolite number for all tabs
    def update_met_idx(self):
        """
        Save metabolite index so we can access it easily later
        """
        if self.updating is False:
            self.met_idx = self.combo_met.currentIndex()

    # Update metabolite dropdowns to account for number of metabolites
    def update_combo_met(self):
        """
        Updates metabolite selector dropbox
        """

        self.updating = True
        self.combo_met.clear()

        # Update dropdown
        n_met = self.spin_n_met.value()
        met_list = [f"Met. {i + 1}" for i in range(n_met)]
        self.combo_met.addItems(met_list)

        # Update param dictionary
        delta_met = n_met - len(self.param_dic["mets"])
        if delta_met > 0:
            for i in range(delta_met):
                self.param_dic["mets"].append(deepcopy(self.param_dic["mets"][0]))
        self.updating = False

    # Update plot selector
    def update_combo_plot(self):
        """
        Function to update plot selector based on current parameters
        """

        # Clear current selector
        self.updating = True
        self.combo_plot.clear()

        # Get new options
        plot_list = ["Waveforms", "2D k-space"]
        if self.param_dic["acq_3d"] is True:
            plot_list.append("3D k-space")
        if hasattr(self, "loc_dic"):
            plot_list.append("Localizer")

        # Update dropdown
        self.combo_plot.addItems(plot_list)
        if hasattr(self, "loc_dic"):
            if self.local_loaded is True:
                self.combo_plot.setCurrentText("Localizer")
                self.local_loaded = False
        self.updating = False

    # Update orientation selector
    def update_combo_pe_dir(self):
        """
        Function to update phase encoding dir based on current parameters
        """
        self.updating = True
        if self.combo_ori.currentText() == "Sagittal":
            self.combo_pe_dir.setItemText(0, "AP")
            self.combo_pe_dir.setItemText(1, "SI")
        elif self.combo_ori.currentText() == "Coronal":
            self.combo_pe_dir.setItemText(0, "RL")
            self.combo_pe_dir.setItemText(1, "SI")
        else:
            self.combo_pe_dir.setItemText(0, "AP")
            self.combo_pe_dir.setItemText(1, "RL")
        self.updating = False

    # Update sequence for plotting
    def update_for_plot(self):
        """
        Creates a new sequence and updates UI and plot
        """
        self.dic_to_seq(no_reps=True)
        self.update_app()

    # Updates to application after creating sequences
    def update_app(self):
        """
        Update UI application based on current parameters
        """

        # Make sure we have current parameters
        self.param_dic = self.seq.create_param_dic()

        # Compute duration
        n_acq = self.param_dic["n_rep"] * self.param_dic["n_avg"]
        self.duration = self.seq.duration()[0] * n_acq

        # Set total sequence duration
        self.dbl_spin_tscan.setValue(np.round(self.duration, 5))

        # Plot k-space
        self.waves = x_epi.compute_k_space(self.plot_seq)
        self.update_combo_plot()
        self.plot()

        # Update UI
        self.dic_to_ui()

        # Run timing check
        status = self.seq.check_timing()
        if any(status) is True:
            self.time_label.setStyleSheet("background-color: green;" "color: white;")
            self.time_label.setText("Timing Passed")
        else:
            self.time_label.setStyleSheet("background-color: red;" "color: white;")
            self.time_label.setText("Timing Error")
            self.time_label.setToolTip(str(status[1]))
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)

    # Calculate cumulative flip angle
    def update_flips(self, no_update=False):
        """
        Compute cumulative flip angles

        Parameters
        ----------
        no_update : bool
           If true, does not change updating status
        """

        if no_update is False:
            self.updating = True

        flip = self.param_dic["mets"][self.met_idx]["flip"]
        n_z = self.param_dic["mets"][self.met_idx]["size"][2]
        n_r = self.param_dic["n_rep"]
        n_a = self.param_dic["n_avg"]
        if self.param_dic["acq_3d"] is True:
            n_z = np.round(self.param_dic["mets"][self.met_idx]["pf_pe2"] * n_z)
            n_acq = n_z * n_r * n_a
            n_vol = n_z
        else:
            n_acq = n_r * n_a
            n_vol = 1
        cum_flip = np.rad2deg(np.arccos(np.power(np.cos(np.deg2rad(flip)), n_acq)))
        vol_flip = np.rad2deg(np.arccos(np.power(np.cos(np.deg2rad(flip)), n_vol)))
        self.line_cum_flip.setText(str(np.round(cum_flip, 1)))
        self.line_vol_flip.setText(str(np.round(vol_flip, 1)))

        if no_update is False:
            self.updating = False

    def save_seq(self):
        """
        Generate Pulseq 'seq' file, json file of parameters, and k-space locations
        """
        save_path = QFileDialog.getSaveFileName(self, "Save Sequence")[0]
        self.dic_to_seq()
        self.seq.write(save_path)
        x_epi.save_k_space(self.plot_seq, save_path)
        self.update_app()
        self.seq.save_params(save_path)

    # Update all the variables that change when selected metabolite change
    def update_met_ui(self):
        """
        Updates metabolite values in interface using current sequence
        """

        # Update interface values based on dictionary
        self.updating = True
        curr_met = self.param_dic["mets"][self.met_idx]
        self.line_grd_path.setText(curr_met["grd_path"])
        self.line_rf_path.setText(curr_met["rf_path"])
        self.line_formula.setText(curr_met["formula"])
        self.combo_use_sinc.setCurrentIndex(curr_met["use_sinc"])
        self.dbl_spin_flip.setValue(curr_met["flip"])
        self.dbl_spin_freq_off.setValue(curr_met["freq_off"])
        self.dbl_spin_sinc_dur.setValue(curr_met["sinc_dur"])
        self.dbl_spin_sinc_frac.setValue(curr_met["sinc_frac"])
        self.dbl_spin_sinc_tbw.setValue(curr_met["sinc_tbw"])
        self.spin_size_ro.setValue(curr_met["size"][0])
        self.spin_size_pe.setValue(curr_met["size"][1])
        self.spin_size_slc.setValue(curr_met["size"][2])
        self.dbl_spin_pf_pe.setValue(curr_met["pf_pe"])
        self.dbl_spin_pf_pe2.setValue(curr_met["pf_pe2"])
        self.line_esp.setText(str(np.round(curr_met["esp"] * 1e3, 2)))
        self.line_b1_max.setText(str(curr_met["b1_max"]))
        self.line_rf_delta.setText(str(curr_met["rf_delta"]))
        self.line_grd_max.setText(str(curr_met["grd_max"]))
        self.line_grd_delta.setText(str(curr_met["grd_delta"]))
        self.check_z_centric.setChecked(curr_met["z_centric"])
        self.line_name.setText(str(curr_met["name"]))
        self.update_flips(no_update=True)
        self.plot()
        self.updating = False

    def load_pars(self):
        """
        Function to load in parameters for json and update plot
        """
        qm = QMessageBox()
        json_path = QFileDialog.getOpenFileName(qm, "Load Parameter File")[0]
        self.load_json(json_path, use_default=False)
        self.update_for_plot()

    def load_localizer(self):
        """
        Function to load in localizer info
        """

        # Convert dicom images into nifti
        qm = QMessageBox()
        dcm_dir = QFileDialog.getExistingDirectory(qm, "Load Localizer DICOM Direcotry")
        _ = sp.run(
            [f"dcm2niix -z y {dcm_dir}"],
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )

        # Loop through json files
        self.loc_dic = {}
        for j_path in glob.glob(f"{dcm_dir}/*.json"):
            # Load in json data
            with open(j_path, "r", encoding="utf-8") as j_id:
                j_data = json.load(j_id)

            # Use json to figure out orientation string
            ori_str = j_data["ImageOrientationText"][0:3].lower()

            # Load in image data
            nii_path = j_path.split(os.extsep, 1)[0] + ".nii.gz"
            img_hdr = nib.load(nii_path)
            img_data = img_hdr.get_fdata()

            # Get orientation codes for each axis
            ax_codes = nib.aff2axcodes(img_hdr.affine)
            ax_codes_neg = nib.aff2axcodes(
                img_hdr.affine
                @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            )

            # Get affine matrixes
            vox_to_scan = img_hdr.affine
            scan_to_vox = np.linalg.inv(vox_to_scan)

            # Add data to dictionary
            self.loc_dic[ori_str] = {
                "hdr": img_hdr,
                "img": img_data,
                "codes+": ax_codes,
                "codes-": ax_codes_neg,
                "vox_to_scan": vox_to_scan,
                "scan_to_vox": scan_to_vox,
            }

        # Define rotation matrices for each orientation
        sag_mat = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )  # phase encode is A/P
        cor_mat = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )  # phase encode is R/L
        tra_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # phase encode is A/P

        # Get axes codes
        sag_codes = nib.aff2axcodes(sag_mat)
        cor_codes = nib.aff2axcodes(cor_mat)
        tra_codes = nib.aff2axcodes(tra_mat)

        # Define dictionary with orientation information
        self.ori_dic = {
            "sag": {"name": "Sagittal", "aff": sag_mat, "codes": sag_codes},
            "cor": {"name": "Coronal", "aff": cor_mat, "codes": cor_codes},
            "tra": {"name": "Transverse", "aff": tra_mat, "codes": tra_codes},
        }

        # Define slices
        self.loc_dic["sag"]["slice"] = 0
        self.loc_dic["cor"]["slice"] = 0
        self.loc_dic["tra"]["slice"] = 0

        # Create primary phase encoding directions
        self.loc_dic["sag"]["pe_prim"] = "AP"
        self.loc_dic["cor"]["pe_prim"] = "RL"
        self.loc_dic["tra"]["pe_prim"] = "AP"
        self.local_loaded = True


def main():
    """
    GUI startup function
    """

    # Setup application
    app = QApplication(sys.argv)
    icon_path = os.path.join(x_epi.RES_DIR, "x_epi_logo.png")
    app.setWindowIcon(QIcon(icon_path))

    # Ask user if they want to load preset parameters
    qm = QMessageBox()
    qm.setText("Parameter Setup")
    qm.addButton("Load Custom", QMessageBox.YesRole)
    qm.addButton("Use Default", QMessageBox.NoRole)
    q_val = qm.exec()

    # Load preset parameters if necessary
    if q_val == 1:
        json_path = None
    else:
        json_path = QFileDialog.getOpenFileName(qm, "Load Parameter File")[0]

    # Load app
    window = MyMainWindow(json_path=json_path)
    window.update_met_ui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
