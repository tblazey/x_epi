#!/usr/bin/python

#Load libraries
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

#Load in simulation data
sim_path = 'pulse_sim.npy'
sim_data = np.load(sim_path)
out_path = 'fig_s2.tiff'

#Extract components
sig = sim_data['sig'][::-1, :]         #invert so that signs match what we see on scanner
freqs = sim_data['freqs'] / 2 / np.pi  #hz        
dists = sim_data['dists'] * 100        #cm
n_freq = freqs.shape[0]
n_dist = dists.shape[0]

#Define metabolites where we should should show resonance
mets = (np.array([184.9, 181, 178.4, 172.8, 162.7]) - 172.8) * 3 * 10.7084
names = ['Lac', 'PyrH', 'Ala', 'Pyr', 'Bic']

#Create gridspec figure
fig = plt.figure(figsize=(12, 10.5))
gs = GridSpec(3, 3, figure=fig, wspace=0,
              width_ratios=[1, 0.25, 1], height_ratios=[1, 0.05, 1])
          
#Plot simulated profile
ax_pro = fig.add_subplot(gs[0, :])
ax_pro.grid()
ax_pro.matshow(np.abs(sig).T, cmap='gray', extent=[freqs[0], freqs[-1], dists[0], dists[-1]], aspect='auto')
ax_pro.xaxis.set_ticks_position('bottom')
ax_pro.set_xlabel('Frequency Offset (Hz)', fontweight='bold', fontsize=14)
ax_pro.set_ylabel('Distance (cm)', fontweight='bold', fontsize=14)
ax_pro.tick_params(axis='both', labelsize=12)

#Add subplots for dist = 0
dist_idx = np.argmin(np.abs(dists))
freq_profile = np.abs(sig[:, dist_idx])
ax_freq = fig.add_subplot(gs[2, 0])
ax_freq.grid()
ax_freq.plot(freqs, freq_profile, c='#006bb6', lw=2)
ax_freq.set_xlabel('Frequency Offset (Hz)', fontweight='bold', fontsize=14, labelpad=10)
ax_freq.set_ylabel('Signal (A.U.)', fontweight='bold', fontsize=14, labelpad=10)

#Add metabolite lines and labels
for met,name in zip(mets, names):

   #Lines for 2D profile
   ax_pro.axvline(met, ls='dashed', color='white', linewidth=2)
   ax_pro.annotate(name, (met - 40, np.max(dists)), (met - 25, np.max(dists) * 1.05),
                   xycoords='data', fontweight='bold', fontsize=14)
                   
   #Lines for 1D profile
   ax_freq.axvline(met, ls='dashed', color='black', linewidth=2)
   ax_freq.annotate(name, (met - 40, 1), (met - 40, 1.075),
                   xycoords='data', fontweight='bold', fontsize=12, rotation=45)

#And subplot for freq = 0
freq_idx = np.argmin(np.abs(freqs))
dist_profile = np.abs(sig[freq_idx, :])
ax_dist = fig.add_subplot(gs[2, 2])
ax_dist.grid()
ax_dist.plot(dists, dist_profile, c='#006bb6', lw=2)
ax_dist.set_xlabel('Distance (cm)', fontweight='bold', fontsize=14, labelpad=10)
ax_dist.set_ylabel('Signal (A.U.)', fontweight='bold', fontsize=14, labelpad=10)

#Subfigure annotations
plt.annotate("A)", fontsize=24, xy=(0.035, 0.83), xycoords='figure fraction', 
             fontweight='bold')
plt.annotate("B)", fontsize=24, xy=(0.035, 0.405), xycoords='figure fraction', 
             fontweight='bold')    
plt.annotate("C)", fontsize=24, xy=(0.47, 0.405), xycoords='figure fraction', 
             fontweight='bold')  

#Find distance fwhm(works for single peak without noise)
freq_left = interp.interp1d(freq_profile[0:freq_idx], freqs[0:freq_idx])(0.5)
freq_right = interp.interp1d(freq_profile[freq_idx::], freqs[freq_idx::])(0.5)
freq_fwhm = freq_right - freq_left
print(f'Frequency FWHM: {freq_fwhm} Hz')

#Find distance fwhm(works for single peak without noise)
dist_left = interp.interp1d(dist_profile[0:dist_idx], dists[0:dist_idx])(0.5)
dist_right = interp.interp1d(dist_profile[dist_idx::], dists[dist_idx::])(0.5)
dist_fwhm = dist_right - dist_left
print(f'Distance FWHM: {dist_fwhm} cm')

plt.savefig(out_path, dpi=225, bbox_inches='tight')
plt.close('all')
