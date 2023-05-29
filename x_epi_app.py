#Load libraries
import numpy as np  
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QMessageBox, QPushButton
from PyQt5 import QtCore
import sys 
import re
import x_epi
from x_epi_ui import Ui_MainWindow
import json

class ScalarFormatterForceFormat(ScalarFormatter):
   #https://stackoverflow.com/questions/42142144/displaying-first-decimal-digit-in-scientific-notation-in-matplotlib/42156450#42156450#
   def __init__(self):
      super().__init__()
      self.set_powerlimits((0, 0))
   def _set_format(self):
      self.format = "%+1.1f"

basedir = os.path.dirname(__file__)
class MyMainWindow(QMainWindow, Ui_MainWindow):
   def __init__(self, json_path=None):
      super().__init__()
      
      #Add UI to class
      self.setupUi(self)
      self.Tabs.setCurrentIndex(0)  
      
      #Class variables that are modified by UI
      try:
      
         #Load in json data
         fid = open(json_path, 'r')
         json_data = json.load(fid)

         #Set values
         self.flips = json_data['flips']
         self.rf_paths = json_data['rf_paths']
         self.grd_paths = json_data['grd_paths']
         self.rf_types = json_data['rf_types']
         self.max_grd = json_data['max_grd']
         self.max_b1 = json_data['max_b1']
         self.rf_delta = json_data['rf_delta']
         self.grd_delta = json_data['grd_delta']
         self.freq_offset = json_data['freq_offset']
         self.sinc_dur = json_data['sinc_dur']
         self.sinc_cf = json_data['sinc_cf']
         self.sinc_tbw = json_data['sinc_tbw']
         self.gamma = json_data['gamma']
         self.size = np.array(json_data['size'])
         self.pf = np.array(json_data['pf'])
         self.grd_forms = json_data['grd_forms']
         self.spin_fov_x.setValue(json_data['fov'][0])
         self.spin_fov_y.setValue(json_data['fov'][1])
         self.spin_fov_z.setValue(json_data['fov'][2])
         self.spin_band.setValue(json_data['bandwidth'])
         self.spin_avg.setValue(json_data["n_avg"])
         self.spin_rep.setValue(json_data["n_rep"])
         self.dbl_spin_tr.setValue(json_data["tr"])
         self.dbl_spin_tv.setValue(json_data["tv"])
         self.dbl_spin_ts.setValue(json_data["ts"])
         self.spin_met.setValue(json_data["n_met"])
         self.dbl_spin_grd_limit.setValue(json_data["grd_limit"])
         self.dbl_spin_slew_limit.setValue(json_data["slew_limit"])
         self.dbl_spin_ringdown.setValue(json_data["ringdown"])
         self.dbl_spin_dead_time.setValue(json_data["dead_time"])
         self.dbl_spin_b0.setValue(json_data["b0"])
         self.check_spectra_start.setChecked(bool(json_data["spectra_start"]))
         self.check_spectra_end.setChecked(bool(json_data["spectra_end"]))
         self.spin_spectra_points.setValue(json_data["spectra_points"])
         self.spin_spectra_band.setValue(json_data["spectra_band"])
         self.dbl_spin_spectra_flip.setValue(json_data["spectra_flip"])
         self.combo_slice_axis_sel.setCurrentText(json_data["slice_axis"])
         self.check_read_alt.setChecked(bool(json_data["read_alt"]))
         self.check_phase_alt.setChecked(bool(json_data["phase_alt"]))
         self.check_slice_alt.setChecked(bool(json_data["slice_alt"]))
         self.update_met_sel()

         #Set radiobuttons
         groups = [self.button_group_readout, self.button_group_mode,
                   self.button_group_phase, self.button_group_slice, 
                   self.button_group_spoil, self.button_group_skip,
                   self.button_group_scale]
         vals = [json_data['readout'], json_data['mode'], json_data['phase'],
                 json_data['slice'], json_data['spoil'], json_data['skip'],
                 json_data['scale']]
         for group, val in zip(groups, vals):
            for elem in group.buttons():
               if elem.text() == val:
                  elem.setChecked(True)
               
      except Exception as e:
         print(e)
         #Set default values
         self.flips = [20, 90, 90, 90, 90]
         self.rf_paths = [os.path.join(basedir, 'ssrf', 'siemens_pyr_plateau_slab.RF'),
                          os.path.join(basedir, 'ssrf', 'siemens_lac_plateau_slab.RF'),
                          os.path.join(basedir, 'ssrf', 'siemens_bic_plateau_slab.RF'),
                          '', '']
         self.grd_paths = [os.path.join(basedir, 'ssrf', 'siemens_pyr_plateau_slab.GRD'),
                           os.path.join(basedir, 'ssrf', 'siemens_lac_plateau_slab.GRD'),
                           os.path.join(basedir, 'ssrf', 'siemens_bic_plateau_slab.GRD'),
                           '', '']
         self.rf_types = [0, 0, 0, 0, 0]
         self.max_grd = [1.4898, 1.4751, 1.4751, 0, 0]
         self.max_b1 = [0.1146, 0.1247, 0.1301, 0, 0]
         self.rf_delta = [4E-6, 4E-6, 4E-6, 4E-6, 4E-6]
         self.grd_delta = [4E-6, 4E-6, 4E-6, 4E-6, 4E-6]
      
         self.freq_offset = [0, 0, 0, 0, 0]
         self.sinc_dur = [4E-3, 4E-3, 4E-3, 4E-3, 4E-3]
         self.sinc_cf = [0.5, 0.5, 0.5, 0.5, 0.5]
         self.sinc_tbw = [4, 4, 4, 4, 4]
         self.gamma = 10.7084E6
         self.size = np.array([[32, 32, 32], [24, 24, 24], [16, 16, 16],
                               [8, 8, 8], [8, 8, 8]])
         self.pf = np.array([[1, 0.8], [1, 1], [1, 1], [1, 1], [1, 1]])
         self.grd_forms = ['1', '1', '1', '1', '1']
      self.met_idx = 0
      self.esp = [0, 0, 0, 0, 0]
      
      #Prep for figure
      self.figure = plt.figure(figsize=(10, 5.5))
      self.canvas = FigureCanvasQTAgg(self.figure)
      self.plot_layout = QVBoxLayout()
      self.plot_layout.addWidget(self.canvas, alignment=QtCore.Qt.AlignTop)
      self.centralwidget.setLayout(self.plot_layout)
      self.update_time()
      self.updating = False
      
      #Updates for spectra
      self.button_group_spectra.buttonToggled.connect(self.spectra_update)
      
      #If number of metabolites is changed
      self.spin_met.valueChanged.connect(self.update_met_sel)
               
      #If metabolite selection is changed on plot page        
      list(map(self.combo_met_sel.currentIndexChanged.connect,
               [self.update_met_idx, self.update_met_vals]))
      
      #If flip angle is changed
      list(map(self.dbl_spin_flip.valueChanged.connect,
               [self.record_flip, self.update_flips]))
      
      #If load rf/gradient buttons are pressed
      self.button_load_rf.clicked.connect(self.load_rf)
      self.button_load_grd.clicked.connect(self.load_grd)
      
      #If time update/save buttons are pressed
      self.button_update.clicked.connect(self.update_time)
      self.button_save.clicked.connect(self.save_seq)
      
      #If readout is changed
      self.button_group_readout.buttonToggled.connect(self.readout_update)
      
      #If nucleus is changed
      self.combo_nucleus.currentIndexChanged.connect(self.update_gamma)
      
      #If pulse type is changed
      self.combo_pulse_type_sel.currentIndexChanged.connect(self.update_rf_type)
      
      #If image grid items are changed
      self.spin_size_x.textChanged.connect(self.record_size)
      self.spin_size_y.textChanged.connect(self.record_size)
      self.spin_size_z.textChanged.connect(self.record_size)
      
      #Flip angle updates
      self.spin_size_z.textChanged.connect(self.update_flips)
      self.spin_rep.textChanged.connect(self.update_flips)
      self.spin_avg.textChanged.connect(self.update_flips)
      
      #If partial fourier options are changed
      self.dbl_spin_pf_y.textChanged.connect(self.record_pf)
      self.dbl_spin_pf_z.textChanged.connect(self.record_pf)
      
      #If frequency offset is changed
      self.dbl_spin_freq.textChanged.connect(self.record_freq)
      
      #If gradient scale formula
      self.line_grd_form.textChanged.connect(self.record_form)
      
      #If plot type is changed
      self.combo_plot_sel.currentIndexChanged.connect(self.plot_k_space)
      
      #Record sinc pulse updates
      self.dbl_spin_sinc_dur.textChanged.connect(self.record_sinc)
      self.dbl_spin_sinc_cf.textChanged.connect(self.record_sinc)
      self.dbl_spin_sinc_tbw.textChanged.connect(self.record_sinc)

   def readout_update(self):
      if self.button_group_readout.checkedButton().text() == 'Flyback':
         state = True
      else:
         state = False
      self.radio_skip_off.setHidden(state)
      self.radio_skip_on.setHidden(state)
      self.label_skip_echo.setHidden(state)
         
   def spectra_update(self):
      if self.check_spectra_start.isChecked() or self.check_spectra_end.isChecked():
         state = True
      else:
         state = False
      self.spin_spectra_points.setEnabled(state)
      self.spin_spectra_band.setEnabled(state)
      self.dbl_spin_spectra_flip.setEnabled(state)
      
   def update_rf_type(self):
      self.rf_types[self.met_idx] = self.combo_pulse_type_sel.currentIndex()
      if self.combo_pulse_type_sel.currentIndex() == 0:
         self.stacked_pulse_type.setCurrentIndex(0)
      else:
         self.stacked_pulse_type.setCurrentIndex(1)
        
   def plot_k_space(self):
      self.figure.clear()

      #Make sure we don't have to update plot
      met = self.met_idx
      if len(self.waves[2]) <= met:
         self.update_time()

      #Determine which plot to do
      plot_type = self.combo_plot_sel.currentText()
      if plot_type == "2D k-space" or plot_type == "3D k-space":
         
         if plot_type == "2D k-space":
            ax = self.figure.add_subplot()
            ax.grid()
            ax.plot(self.waves[2][met][0, :], self.waves[2][met][1, :])
            ax.scatter(self.waves[3][met][0, :], self.waves[3][met][1, :], c='red', s=10)
            ax.axis('equal')
         else:
            ax = self.figure.add_subplot(projection='3d')
            ax.set_zlabel(r'$k_z$ ($mm^{-1}$)')
            ax.plot(self.waves[0][met][0, :], self.waves[0][met][1, :],
                    self.waves[0][met][2, :])
            ax.scatter(self.waves[1][met][0, :], self.waves[1][met][1, :],
                       self.waves[1][met][2, :], c='red', s=2)
         
         #Common k-space plot optionsn   
         ax.set_xlabel(r'$k_x$ ($mm^{-1}$)')
         ax.set_ylabel(r'$k_y$ ($mm^{-1}$)')
         
      else:
      
         #Common options
         t = np.arange(self.waves[4][self.met_idx].shape[1]) * \
                       self.plot_seq.system.grad_raster_time * 1E3

         #Magnitude plot
         ax_mag = self.figure.add_subplot(5, 1, 1)
         ax_mag.grid()
         ax_mag.plot(t, np.abs(self.waves[5][self.met_idx]), linewidth=0.5)
         ax_mag.set_ylabel('Mag.\n(Hz)', fontweight='bold', rotation=0)
         ax_mag.xaxis.set_ticklabels([])
         ax_mag.yaxis.set_label_coords(-.1, 0.3)
         ax_mag.yaxis.set_major_formatter(ScalarFormatterForceFormat())
         
         #Phase plot
         ax_phase = self.figure.add_subplot(5, 1, 2)
         ax_phase.grid()
         ax_phase.plot(t, np.angle(self.waves[5][self.met_idx]), linewidth=0.5)
         ax_phase.set_ylabel('Phase\n(rad)', fontweight='bold', rotation=0)
         ax_phase.xaxis.set_ticklabels([])
         ax_phase.yaxis.set_label_coords(-.1, 0.3)
         ax_phase.yaxis.set_major_formatter(ScalarFormatterForceFormat())
             
         #X gradient plot
         ax_x = self.figure.add_subplot(5, 1, 3)
         ax_x.grid()
         ax_x.plot(t, self.waves[4][self.met_idx][0, :], linewidth=0.5)
         ax_x.set_ylabel('Gx\n(Hz/m)', fontweight='bold', rotation=0)
         ax_x.xaxis.set_ticklabels([])
         ax_x.yaxis.set_label_coords(-.1, 0.3)
         ax_x.yaxis.set_major_formatter(ScalarFormatterForceFormat())
         
         #Y gradient plot
         ax_y = self.figure.add_subplot(5, 1, 4)
         ax_y.grid()
         ax_y.plot(t, self.waves[4][self.met_idx][1, :], linewidth=0.5)
         ax_y.set_ylabel('Gy\n(Hz/m)', fontweight='bold', rotation=0)
         ax_y.xaxis.set_ticklabels([])
         ax_y.yaxis.set_label_coords(-.1, 0.3)
         ax_y.yaxis.set_major_formatter(ScalarFormatterForceFormat())
         
         #Z gradient plot
         ax_z = self.figure.add_subplot(5, 1, 5)
         ax_z.grid()
         ax_z.plot(t, self.waves[4][self.met_idx][2, :], linewidth=0.5)
         ax_z.set_xlabel('Time (ms)', fontweight='bold')
         ax_z.set_ylabel('Gz\n(Hz/m)', fontweight='bold', rotation=0)
         ax_z.yaxis.set_label_coords(-.1, 0.3)
         ax_z.yaxis.set_major_formatter(ScalarFormatterForceFormat())
         
         #Adjust spacing
         self.figure.subplots_adjust(hspace=0.1)
      
      plt.tight_layout()   
      self.canvas.draw()
        
   #Change metabolite number for all tabs
   def update_met_idx(self):
      self.met_idx = self.combo_met_sel.currentIndex()
      
   #Update metabolite number dropdowns to account for number of metabolites
   def update_met_sel(self):
      self.combo_met_sel.clear()
      n_met = self.spin_met.value()
      met_list = ['Met. %i'%(i + 1) for i in range(n_met)]
      self.combo_met_sel.addItems(met_list)
      
   #Save flip angle to class variable
   def record_flip(self):
      if self.updating is False:
         self.flips[self.met_idx] = self.dbl_spin_flip.value()
   
   #Save sinc pulse data
   def record_sinc(self):
      if self.updating is False:
         self.sinc_dur[self.met_idx] = self.dbl_spin_sinc_dur.value() / 1E3
         self.sinc_cf[self.met_idx] = self.dbl_spin_sinc_cf.value()
         self.sinc_tbw[self.met_idx] = self.dbl_spin_sinc_tbw.value()
   
   #Save flip angle to class variable
   def record_freq(self):
      if self.updating is False:
         self.freq_offset[self.met_idx] = self.dbl_spin_freq.value()
   
   #Save image grid sizes for current metabolites   
   def record_size(self):
      if self.updating is False:
         self.size[self.met_idx, :] = [self.spin_size_x.value(),
                                       self.spin_size_y.value(),
                                       self.spin_size_z.value()]
   
   #Save formula
   def record_form(self):
      if self.updating is False:
         self.grd_forms[self.met_idx] = self.line_grd_form.text()
   
   #Save partial fourier fractions
   def record_pf(self):
      if self.updating is False:
         self.pf[self.met_idx, :] = [self.dbl_spin_pf_y.value(),
                                     self.dbl_spin_pf_z.value()]                                     
   
   #Update sequence time
   def update_time(self):
      self.seq, self.plot_seq = x_epi.create_seq(self, only_plot=True)
      self.update_app()
   
   #Updates to application after creating sequences
   def update_app(self):
      n_acq = self.spin_rep.value() * self.spin_avg.value()
      self.duration = self.seq.duration()[0] * n_acq
      self.dbl_spin_tscan.setValue(np.round(self.duration, 5))
      self.line_esp.setText(str(np.round(self.esp[self.met_idx] * 1E3, 2)))
      self.waves = x_epi.compute_k_space(self.plot_seq, self.spin_met.value())
      self.plot_k_space()
      status = self.seq.check_timing()
      
      if status[0] == True:
         self.time_label.setStyleSheet("background-color: green;"
                                       "color: white;")
         self.time_label.setText('Timing Passed')
      else:
         self.time_label.setStyleSheet("background-color: red;"
                                       "color: white;")
         self.time_label.setText('Timing Error')
         self.time_label.setToolTip(str(status[1]))
      self.time_label.setAlignment(QtCore.Qt.AlignCenter)

   #Calculate cumulative flip angle
   def update_flips(self):
       flip = self.dbl_spin_flip.value()
       n_z = self.size[self.met_idx, 2]
       n_r = self.spin_rep.value()
       n_a = self.spin_avg.value()
       n_acq = n_z * n_r * n_a
       cum_flip = np.rad2deg(np.arccos(np.power(np.cos(np.deg2rad(flip)), n_acq)))
       vol_flip =  np.rad2deg(np.arccos(np.power(np.cos(np.deg2rad(flip)), n_z)))
       self.line_cum_flip.setText(str(np.round(cum_flip, 1)))
       self.line_vol_flip.setText(str(np.round(vol_flip, 1)))
       
   def save_seq(self):
      save_path = QFileDialog.getSaveFileName(self, 'Save Sequence')[0]  
      self.seq, self.plot_seq = x_epi.create_seq(self, return_plot=True)
      self.seq.write(save_path)
      self.save_params(save_path)
      self.update_app()
   
   def save_params(self, save_path):
      out_dict = {
                   "flips":self.flips,
                   "rf_paths":self.rf_paths,
                   "grd_paths":self.grd_paths,
                   "rf_types":self.rf_types,
                   "max_grd":self.max_grd,
                   "max_b1":self.max_b1,
                   "rf_delta":self.rf_delta,
                   "grd_delta":self.grd_delta,
                   "freq_offset":self.freq_offset,
                   "sinc_dur":self.sinc_dur,
                   "sinc_cf":self.sinc_cf,
                   "sinc_tbw":self.sinc_tbw,
                   "gamma":self.gamma,
                   "size":self.size.tolist(),
                   "pf":self.pf.tolist(),
                   "grd_forms":self.grd_forms,
                   "fov":[self.spin_fov_x.value(), self.spin_fov_y.value(), 
                         self.spin_fov_z.value()],
                   "bandwidth":self.spin_band.value(),
                   "n_avg":self.spin_avg.value(),
                   "n_rep":self.spin_rep.value(),
                   "tr":self.dbl_spin_tr.value(),
                   "tv":self.dbl_spin_tv.value(),
                   "ts":self.dbl_spin_ts.value(),
                   "n_met":self.spin_met.value(),
                   "grd_limit":self.dbl_spin_grd_limit.value(),
                   "slew_limit":self.dbl_spin_slew_limit.value(),
                   "ringdown":self.dbl_spin_ringdown.value(),
                   "dead_time":self.dbl_spin_dead_time.value(),
                   "b0":self.dbl_spin_b0.value(),
                   "spectra_start":self.check_spectra_start.isChecked(),
                   "spectra_end":self.check_spectra_end.isChecked(),
                   "spectra_points":self.spin_spectra_points.value(),
                   "spectra_band":self.spin_spectra_band.value(),
                   "spectra_flip":self.dbl_spin_spectra_flip.value(),
                   "slice_axis":self.combo_slice_axis_sel.currentText(),
                   "readout":self.button_group_readout.checkedButton().text(),
                   "mode":self.button_group_mode.checkedButton().text(),
                   "phase":self.button_group_phase.checkedButton().text(),
                   "slice":self.button_group_slice.checkedButton().text(),
                   "spoil":self.button_group_spoil.checkedButton().text(),
                   "skip":self.button_group_skip.checkedButton().text(),
                   "read_alt":self.check_read_alt.isChecked(),
                   "phase_alt":self.check_phase_alt.isChecked(),
                   "slice_alt":self.check_slice_alt.isChecked(),
                   "scale":self.button_group_scale.checkedButton().text()   
                  }

      with open('%s.json'%(save_path), "w") as fid:
         json.dump(out_dict, fid, indent=2)
      
   #Load gradient waveforms
   def load_grd(self):
      grd_path = QFileDialog.getOpenFileName(self, 'Load Gradient File')[0]  
      
      #Get max gradient strength for scaling
      with open(grd_path) as fid:
         grd_txt = fid.read()
      max_grd = float(re.search('Max Gradient Strength = .*', grd_txt)[0].split()[4])
      delta = float(re.search('Resolution = .*', grd_txt)[0].split()[2])

      #Save path and maximum gradient
      self.grd_paths[self.met_idx] = grd_path
      self.max_grd[self.met_idx] = max_grd
      self.grd_delta[self.met_idx] = delta * 1E-6
      self.line_grd_path.setText(self.grd_paths[self.met_idx])
      self.line_max_grd.setText(str(self.max_grd[self.met_idx]))
      
   #Load rf waveforms
   def load_rf(self):
      rf_path = QFileDialog.getOpenFileName(self, 'Load RF File')[0]  
      
      #Get max b1 for scaling
      with open(rf_path) as fid:
         rf_txt = fid.read()
      max_b1 = float(re.search('Max B1 = .*', rf_txt)[0].split()[3])
      delta = float(re.search('Resolution = .*', rf_txt)[0].split()[2])
      
      #Save path and maximum gradient
      self.rf_paths[self.met_idx] = rf_path
      self.max_b1[self.met_idx] = max_b1
      self.rf_delta[self.met_idx] = delta * 1E-6
      self.line_rf_path.setText(self.rf_paths[self.met_idx])
      self.line_max_b1.setText(str(self.max_b1[self.met_idx]))
     
   #Update all the variables that change when selected metabolite change 
   def update_met_vals(self):
      self.updating = True
      self.combo_met_sel.setCurrentIndex(self.met_idx)
      self.combo_pulse_type_sel.setCurrentIndex(self.rf_types[self.met_idx])
      self.line_rf_path.setText(self.rf_paths[self.met_idx])
      self.line_grd_path.setText(self.grd_paths[self.met_idx])
      self.line_max_b1.setText(str(self.max_b1[self.met_idx]))
      self.line_max_grd.setText(str(self.max_grd[self.met_idx]))
      self.line_rf_delta.setText(str(self.rf_delta[self.met_idx] * 1E6))
      self.line_grd_delta.setText(str(self.grd_delta[self.met_idx] * 1E6))
      self.dbl_spin_flip.setValue(self.flips[self.met_idx])
      self.spin_size_x.setValue(self.size[self.met_idx, 0])
      self.spin_size_y.setValue(self.size[self.met_idx, 1])
      self.spin_size_z.setValue(self.size[self.met_idx, 2])
      self.dbl_spin_pf_y.setValue(self.pf[self.met_idx, 0])
      self.dbl_spin_pf_z.setValue(self.pf[self.met_idx, 1])
      self.line_esp.setText(str(np.round(self.esp[self.met_idx] * 1E3, 2)))
      self.dbl_spin_freq.setValue(self.freq_offset[self.met_idx])
      self.dbl_spin_sinc_dur.setValue(self.sinc_dur[self.met_idx] * 1E3)
      self.dbl_spin_sinc_cf.setValue(self.sinc_cf[self.met_idx])
      self.dbl_spin_sinc_tbw.setValue(self.sinc_tbw[self.met_idx])
      self.line_grd_form.setText(self.grd_forms[self.met_idx])
      self.update_flips()
      self.plot_k_space()
      self.updating = False
  
   #Update gyromagnetic ratio
   def update_gamma(self):
      match self.combo_nucleus.currentText():
         case "13C":
            self.gamma = 10.7084E6
         case "1H":
            self.gamma = 42.57638474E6
         case "2H":
            self.gamma = 6.536E6
         case "15N":
            self.gamma = -4.316E6
         case "17O":
            self.gamma = -5.722E6
         case "31P":
            self.gamma = 17.235E6
         case "19F":
            self.gamma = 40.078E6
         case "23Na":
            self.gamma = 11.262E6
         case "129Xe":
            self.gamma = -11.777E6
         case _:
            raise ValueError('Unknown nucleus')
            
if __name__ == '__main__':

   #Setup application
   app = QApplication(sys.argv)
   
   #Ask user if they want to load preset parameters
   qm = QMessageBox()
   qm.setText("Parameter Setup")
   qm.addButton('Load Custom', QMessageBox.YesRole)
   qm.addButton('Use Default', QMessageBox.NoRole)
   q_val = qm.exec()
   
   #Load preset parameters if necessary
   if q_val == 1:
      json_path = None
   else:
      json_path = QFileDialog.getOpenFileName(qm, 'Load Parameter File')[0]
      
   #Load app
   window = MyMainWindow(json_path=json_path)
   window.update_met_vals()
   window.show()
   sys.exit(app.exec_())
