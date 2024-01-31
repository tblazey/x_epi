#!/us/bin/python

#Load libs
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import interp1d

#Define data location
pyr_path = '../fig_4/single_echo_pyr_ud_mag.nii.gz'
lac_path = '../fig_4/single_echo_lac_ud_mag.nii.gz'
anat_path = '../fig_4/anat_masked.nii.gz'

#Load in met images
pyr_hdr = nib.load(pyr_path)
lac_hdr = nib.load(lac_path)
anat_hdr = nib.load(anat_path)
pyr_img = pyr_hdr.get_fdata().squeeze()
lac_img = lac_hdr.get_fdata().squeeze()
anat_img = anat_hdr.get_fdata().squeeze()

anat_idx = np.argwhere(anat_hdr.affine[:, 2] != 0)[0][0]
anat_delta = anat_hdr.affine[anat_idx, 2]
anat_zero = anat_hdr.affine[anat_idx, 3]
pyr_idx = np.argwhere(pyr_hdr.affine[:, 2] != 0)[0][0]
pyr_delta = pyr_hdr.affine[pyr_idx, 2]
pyr_zero = pyr_hdr.affine[pyr_idx, 3]
anat_y = np.arange(anat_hdr.shape[2])
anat_x = anat_y * anat_delta + anat_zero

#Zoom lactate image so we have same resolution
lac_zoom = zoom(lac_img, [1.5, 1.5, 1.5, 1], grid_mode=False, order=3)

#Normalize metabolite images
pyr_min, pyr_max = np.percentile(pyr_img, [0.01, 99.99])
lac_min, lac_max = np.percentile(lac_zoom, [0.01, 99.99])
anat_min, anat_max = np.percentile(anat_img, [0.01, 99.99])

#Create gridspec figure
fig = plt.figure(figsize=(18, 9))
gs_widths = np.concatenate((np.tile([1, 1, 0.1], 5), [1]))
gs_widths[14] = 0.25
gs = GridSpec(6, 16, figure=fig, wspace=0,
              width_ratios=gs_widths,
              height_ratios=[0.1, 1, 1, 1, 1, 1])

#Add data to plot
skip = 3
for i in range(5):
   for j in range(5):
      ax_pyr = fig.add_subplot(gs[i + 1, j * skip])
      ax_lac = fig.add_subplot(gs[i + 1, j * skip + 1])
      ax_anat = fig.add_subplot(gs[i + 1, -1])
      
      #Figure out nearest slice
      slc_idx =  2 + i * 2
      slc_coord = pyr_zero + pyr_delta * slc_idx
      ref_idx = np.int32(interp1d(anat_x, anat_y, kind='nearest')(slc_coord))
      
      
      ax_pyr.matshow(pyr_img[:, :, slc_idx, j * 4], cmap='gray', vmin=pyr_min,
                     vmax=pyr_max, origin='lower')
      ax_lac.matshow(lac_zoom[:, :, slc_idx, j * 4], cmap='gray', vmin=lac_min,
                     vmax=lac_max, origin='lower')
      ax_anat.matshow(anat_img[:, :, ref_idx].T, cmap='gray', origin='lower',
                      vmin=anat_min, vmax=anat_max)              
    
      for ax in [ax_pyr, ax_lac, ax_anat]:
        ax.set_aspect('equal')
        ax.axis('off')             


#Add axes annotations
ax_pyr.annotate('', xy=(0.094, 0), xytext=(0.094, 0.94), xycoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05))
ax_pyr.annotate('', xy=(0.865, 0.89), xytext=(0.045, 0.89), xycoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05))

#Add axes label annotations
ax_pyr.annotate('Time (s)', fontweight='bold', xy=(0.465, 0.97),
                xycoords='figure fraction', fontsize=22)
ax_pyr.annotate('Slice (mm)', fontweight='bold', xy=(0.03, 0.4),
                xycoords='figure fraction', fontsize=22, rotation=90)
                
#Metabolite label annotations
plt.annotate("}", fontsize=36, xy=(0.142, 0.075), xycoords='figure fraction',
             rotation=270)
plt.annotate("Pyruvate", fontsize=16, xy=(0.126, 0.05), xycoords='figure fraction', 
             fontweight='bold')
plt.annotate("}", fontsize=36, xy=(0.2075, 0.075), xycoords='figure fraction',
             rotation=270)
plt.annotate("Lactate", fontsize=16, xy=(0.20, 0.05), xycoords='figure fraction', 
             fontweight='bold')                                

#Tick mark annotations
for i in range(5):    
   ax_pyr.annotate(f'{i * 12}', xy=(0.192 + 0.14 * i, 0.875),
                   xytext=(0.192 + 0.14 * i, 0.915),
                   xycoords='figure fraction',
                   horizontalalignment="center", fontsize=16,
                   arrowprops=dict(arrowstyle = '-', linewidth=2.25))
   ax_pyr.annotate(f'{i * 20 - 40}', xy=(0.105, 0.79 - 0.155 * i),
                   xytext=(0.0675, 0.79 - 0.155 * i),
                   xycoords='figure fraction',
                   verticalalignment="center", fontsize=16,
                   rotation=90,
                   arrowprops=dict(arrowstyle = '-', linewidth=2.25))

#Save figue
plt.savefig('fig_s5.tiff', dpi=185)
plt.close('all')
