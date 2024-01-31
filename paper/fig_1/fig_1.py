#!/usr/bin/python

#Libs
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np 
from scipy.interpolate import interp1d
from x_epi import XSeq

#Create sequence
seq = XSeq(
    symm_ro=True, acq_3d=True, max_grad=60, max_slew=120, ori='Coronal', pe_dir='RL'
)
seq.add_met(name='pyr', size=[24, 24, 24], pf_pe2=0.66, flip=2.5)
seq.add_met(name='lac', size=[18, 18, 18], pf_pe2=0.88, flip=27, freq_off=385.5)
seq.add_met(name='bic', size=[16, 16, 16], flip=27, freq_off=-325)
seq.create_seq()
seq.save_params('three_met')
seq.write('three_met')

#Get waveforms
wave_data, _, _, _, _ = seq.waveforms_and_times(append_RF=True)

t = np.arange(0, 1.2352, 10E-6)
x = interp1d(wave_data[0][0, :], wave_data[0][1, :], fill_value=0, bounds_error=False)(t)
y = interp1d(wave_data[1][0, :], wave_data[1][1, :], fill_value=0, bounds_error=False)(t)
z = interp1d(wave_data[2][0, :], wave_data[2][1, :], fill_value=0, bounds_error=False)(t)
rf_mag = interp1d(
    wave_data[3][0, :].real,
    np.abs(wave_data[3][1, :]),
    fill_value=0,
    bounds_error=False)(t)
rf_pha = interp1d(
    wave_data[3][0, :].real,
    np.angle(wave_data[3][1, :]),
    fill_value=0, 
    bounds_error=False)(t)

n_exc = [16, 16, 16]       #number of excitations per metabolite
cols = ['#006bb6', '#b6006b', '#00b64b']
labs = ['Pyruvate', 'Lactate', 'Bicarbonate']

#Masks for each metabolite
pyr_mask = np.logical_and(t > 0, t < 0.0411605)
lac_mask = np.logical_and(t > 0.6586,  t < 0.6926)
bic_mask = np.logical_and(t >  1.2034, t < 1.2354)
masks = [pyr_mask, lac_mask, bic_mask]

#Make time vectors for each metabolite
t_pyr = t[pyr_mask] * 1E3
t_lac = t[lac_mask] * 1E3
t_bic = t[bic_mask] * 1E3
times = [t_pyr, t_lac, t_bic]

#Get gradient limits
grad_all = np.concatenate((x[pyr_mask], y[pyr_mask], z[pyr_mask]))
grad_min = np.min(grad_all) * 1.1
grad_max = np.max(grad_all) * 1.1

#Create figure
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(5, 1)
plt.subplots_adjust(hspace=0.7)

#Create broken axes
baxs = []
for i in range(5):
   if i == 4:
      d = 0.01
   else:
      d = 0
   bax =  brokenaxes(xlims=((-1, t_pyr[-1] + 1),
                            (t_lac[0] - 1, t_lac[-1] + 1),
                            (t_bic[0] - 1, t_bic[-1] + 1)),
                     subplot_spec=gs[i],
                     wspace=0.05, d=d, tilt=65)
   baxs.append(bax)

#RF magnitude
baxs[0].axhline(0, color='black', lw=0.75)
baxs[0].set_ylabel('Amp. (Hz)', fontweight='bold', fontsize=12, labelpad=35)
baxs[0].big_ax.annotate('RF Mag.', [0.9, 0.65], fontsize=14, fontweight='bold',
                       xycoords='axes fraction')

#RF phase
baxs[1].axhline(0, color='black', lw=0.75)
baxs[1].set_ylim([-3.4, 3.4])
baxs[1].set_ylabel('Phase (rad)', fontweight='bold', fontsize=12, labelpad=35)
baxs[1].big_ax.annotate('RF Phase', [0.9, 1.1], fontsize=14, fontweight='bold',
                       xycoords='axes fraction')

#Z gradient
baxs[2].axhline(0, color='black', lw=0.75)
baxs[2].set_yticks([-2E5, -1E5, 0, 1E5])
baxs[2].set_ylabel('Amp. (Hz/m)', fontweight='bold', fontsize=12, labelpad=35)
baxs[2].ticklabel_format(axis='y', style='sci', scilimits=[-3, 3])
baxs[2].big_ax.annotate('Z Grad.', [0.9, 1.3], fontsize=14, fontweight='bold',
                       xycoords='axes fraction')                       

#Y gradient
baxs[3].axhline(0, color='black', lw=0.75)
baxs[3].set_yticks([-2E5, -1E5, 0, 1E5])
baxs[3].set_ylabel('Amp. (Hz/m)', fontweight='bold', fontsize=12, labelpad=35)
baxs[3].ticklabel_format(axis='y', style='sci', scilimits=[-3, 3])
baxs[3].big_ax.annotate('Y Grad.', [0.9, 1.3], fontsize=14, fontweight='bold',
                       xycoords='axes fraction')

#X gradient
baxs[4].axhline(0, color='black', lw=0.75)
baxs[4].set_yticks([-2E5, -1E5, 0, 1E5])
baxs[4].set_ylabel('Amp. (Hz/m)', fontweight='bold', fontsize=12, labelpad=35)
baxs[4].ticklabel_format(axis='y', style='sci', scilimits=[-3, 3])
baxs[4].big_ax.annotate('X Grad.', [0.9, 1.3], fontsize=14, fontweight='bold',
                       xycoords='axes fraction')
baxs[4].set_xlabel('Time (ms)', fontweight='bold', fontsize=14, labelpad=30)


#Add labels indicating number of excitations per metabolite
for i in range(3):
   baxs[4].axs[i].annotate(r'$\bf{n_{exc}}$ = ' + f'{n_exc[i]}', [0.15, 0.85],
                           color=cols[i], xycoords='axes fraction',
                           size=12, fontweight='bold')
       
#Add data to plots
for t, mask, col, lab in zip(times, masks, cols, labs):
   baxs[0].plot(t, rf_mag[mask], color=col, lw=1.5, label=lab)
   baxs[1].plot(t, rf_pha[mask], color=col, lw=1.5)
   baxs[2].plot(t, np.real(z[mask]), color=col, lw=1.5)
   baxs[3].plot(t, np.real(y[mask]), color=col, lw=1.5)
   baxs[4].plot(t, np.real(x[mask]), color=col, lw=1.5)

#Format each broken axis  
for bax in baxs:
   for ax in bax.axs:
   
      #Remove x axis ticks from everything but bottom plot
      if bax != baxs[-1]:
         ax.set_xticks([], [])
         ax.spines['top'].set_visible(False)
         ax.spines['bottom'].set_visible(False)
         ax.spines['right'].set_visible(False)
      
      #Remove y axis unit offset label from everything but first broken axis
      if ax != bax.axs[0]:
         plt.setp(ax.yaxis.get_offset_text(), visible=False)            

#Add title
baxs[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.48, 1.7), fontsize=12)

plt.savefig('fig_1.tiff', dpi=225, bbox_inches='tight')
plt.close('all')


