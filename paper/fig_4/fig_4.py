#!/us/bin/python

#Load libs
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

#Ploting constants
pyr_min = 0.1
pyr_max = 1
lac_min = 0.1
lac_max = 1
anat_min = 250
anat_max = 2500
t_samp = 3


#58 65 60
#Define data location
pyr_path = 'single_echo_pyr_ud_mean_mag_on_anat.nii.gz'
lac_path = 'single_echo_lac_ud_mean_mag_on_anat.nii.gz'
anat_path = 'anat_masked.nii.gz'
c1_path = 'c1_on_anat.nii.gz'
c2_path = 'c2_on_anat.nii.gz'

#Load in roi means
roi_names = [ 'single_echo_pyr_ud_mag_on_anat_c1_means.txt',
              'single_echo_pyr_ud_mag_on_anat_c2_means.txt',
              'single_echo_lac_ud_mag_on_anat_c1_means.txt',
              'single_echo_lac_ud_mag_on_anat_c2_means.txt' ]
roi_vals = [ np.loadtxt(name) for name in roi_names ]
roi_t = np.arange(roi_vals[0].shape[0]) * t_samp
pyr_roi_norm = np.max(roi_vals[0:2])
lac_roi_norm = np.max(roi_vals[2::])

#Load in met images
pyr_hdr = nib.load(pyr_path)
lac_hdr = nib.load(lac_path)
pyr = pyr_hdr.get_fdata().squeeze()
lac = lac_hdr.get_fdata().squeeze()

#Normalize metabolite images
pyr /= np.max(pyr)
lac /= np.max(lac)

#Load in anatomical reference
anat_hdr = nib.load(anat_path)
anat = anat_hdr.get_fdata().squeeze()
anat[anat == 0] = np.nan

#Load in roi images
c1_hdr = nib.load(c1_path)
c2_hdr = nib.load(c2_path)
c1 = c1_hdr.get_fdata().squeeze()
c2 = c2_hdr.get_fdata().squeeze()

#Create gridspec figure
fig = plt.figure(figsize=(16, 8.25))
gs = GridSpec(2, 5, figure=fig, wspace=0.2, width_ratios=[1, 0.75, 0.75, 0.3, 1.75])

#Pyruvate sagittal
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.matshow(anat[88, 0:119, 36:103].T, extent=[-59, 59, -33, 33],
             cmap='gray', vmin=anat_min, vmax=anat_max)
pyr_im = ax_1.matshow(pyr[88, 0:119, 36:103].T, extent=[-59, 59, -33, 33], 
                      cmap='plasma', vmin=pyr_min, vmax=pyr_max)
ax_1.images[0].cmap.set_under(alpha=0)
ax_1.images[1].cmap.set_under(alpha=0)
ax_1.axis('off')
ax_1.annotate('A)', (-0.25, 1.55), xycoords='axes fraction', fontweight='bold', fontsize=23)

#Pyruvate axial
ax_2 = fig.add_subplot(gs[0, 1])
ax_2.matshow(np.flipud(anat[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59],
             cmap='gray', vmin=anat_min, vmax=anat_max)
ax_2.matshow(np.flipud(pyr[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59], 
             cmap='plasma', vmin=pyr_min, vmax=pyr_max)
ax_2.images[0].cmap.set_under(alpha=0)
ax_2.images[1].cmap.set_under(alpha=0)
ax_2.axis('off')

#Pyruvate coronal
ax_3 = fig.add_subplot(gs[0, 2])
ax_3.matshow(anat[22:115, 65, 36:103].T, cmap='gray',
             extent=[-46, 46, -33, 33], vmin=anat_min, vmax=anat_max)
ax_3.matshow(pyr[22:115, 65, 36:103].T, cmap='plasma',
             extent=[-46, 46, -33, 33], vmin=pyr_min, vmax=pyr_max)
ax_3.images[0].cmap.set_under(alpha=0)
ax_3.images[1].cmap.set_under(alpha=0)
ax_3.axis('off')

#Lactate sagittal
ax_4 = fig.add_subplot(gs[1, 0])
ax_4.matshow(anat[88, 0:119, 36:103].T, extent=[-59, 59, -33, 33],
             cmap='gray', vmin=anat_min, vmax=anat_max)
lac_im = ax_4.matshow(lac[88, 0:119, 36:103].T, extent=[-59, 59, -33, 33], 
                      cmap='viridis', vmin=lac_min, vmax=lac_max)
ax_4.images[0].cmap.set_under(alpha=0)
ax_4.images[1].cmap.set_under(alpha=0)
ax_4.axis('off')
ax_4.annotate('C)', (-0.25, 1.55), xycoords='axes fraction', fontweight='bold', fontsize=23)

#Lactate axial
ax_5 = fig.add_subplot(gs[1, 1])
ax_5.matshow(np.flipud(anat[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59],
             cmap='gray', vmin=anat_min, vmax=anat_max)
ax_5.matshow(np.flipud(lac[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59], 
             cmap='viridis', vmin=lac_min, vmax=lac_max)
ax_5.images[0].cmap.set_under(alpha=0)
ax_5.images[1].cmap.set_under(alpha=0)
ax_5.axis('off')

#Lactate coronal
ax_6 = fig.add_subplot(gs[1, 2])
ax_6.matshow(anat[22:115, 65, 36:103].T, cmap='gray',
             extent=[-46, 46, -33, 33], vmin=anat_min, vmax=anat_max)
ax_6.matshow(lac[22:115, 65, 36:103].T, cmap='viridis',
             extent=[-46, 46, -33, 33], vmin=lac_min, vmax=lac_max)
ax_6.images[0].cmap.set_under(alpha=0)
ax_6.images[1].cmap.set_under(alpha=0)
ax_6.axis('off')

#Add pyruvate colorbar
ax_pyr_cb = fig.add_axes([0.265, 0.49, 0.2, 0.025])
pyr_cb = fig.colorbar(pyr_im, cax=ax_pyr_cb, orientation='horizontal')
pyr_cb.set_label(label='Pyruvate Sig. (A.U.)',weight='bold')
ax_pyr_cb.xaxis.set_label_position('top')

#Add lactate colorbar
ax_lac_cb = fig.add_axes([0.265, 0.075, 0.2, 0.025])
lac_cb = fig.colorbar(lac_im, cax=ax_lac_cb, orientation='horizontal')
lac_cb.set_label(label='Lactate Sig. (A.U.)',weight='bold')
ax_lac_cb.xaxis.set_label_position('top')

#Add roi image
ax_roi= fig.add_axes([0.77, 0.7, 0.175, 0.175], zorder=1)
ax_roi.matshow(np.flipud(anat[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59],
                cmap='gray', vmin=anat_min, vmax=anat_max)
ax_roi.matshow(np.flipud(c1[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59],
                cmap='Reds', vmin=0.5, vmax=0.9)
ax_roi.matshow(np.flipud(c2[22:115, 0:119, 60].T), extent=[-46, 46, -59, 59],
                cmap='Blues', vmin=0.5, vmax=0.9)
ax_roi.images[1].cmap.set_over('#006bb6')
ax_roi.images[2].cmap.set_over('#b6006b')
for img in ax_roi.images:
   img.cmap.set_under(alpha=0)
ax_roi.axis('off')

#Compartment #1
ax_6 = fig.add_subplot(gs[0, 4])
ax_6.grid()
ax_6.plot(roi_t, roi_vals[0] / pyr_roi_norm, label='Yes', c='#006bb6', lw=3)
ax_6.plot(roi_t, roi_vals[1] / pyr_roi_norm, label='No', c='#b6006b', lw=3)
ax_6.scatter(roi_t, roi_vals[0] / pyr_roi_norm, s=50, c='#006bb6')
ax_6.scatter(roi_t, roi_vals[1] / pyr_roi_norm, s=50, c='#b6006b')
#ax_6.set_xlabel('Time (s)', fontweight='bold')
ax_6.set_ylabel('Pyruvate Sig. (A.U.)', fontweight='bold')
ax_6.legend(bbox_to_anchor=(0.775, 1.14), ncol=2)
ax_6.annotate('LDH/NADH', (0.1, 1.055), xycoords='axes fraction', fontweight='bold', fontsize=12)
ax_6.annotate('B)', (-0.25, 1), xycoords='axes fraction', fontweight='bold', fontsize=23)

#Compartment #1
ax_7 = fig.add_subplot(gs[1, 4])
ax_7.grid()
ax_7.plot(roi_t, roi_vals[2] / lac_roi_norm, c='#006bb6', lw=3)
ax_7.plot(roi_t, roi_vals[3] / lac_roi_norm, c='#b6006b', lw=3)
ax_7.scatter(roi_t, roi_vals[2] / lac_roi_norm, s=50, c='#006bb6')
ax_7.scatter(roi_t, roi_vals[3] / lac_roi_norm, s=50, c='#b6006b')
ax_7.set_xlabel('Time (s)', fontweight='bold')
ax_7.set_ylabel('Lactate Sig. (A.U.)', fontweight='bold')
ax_7.annotate('D)', (-0.25, 1), xycoords='axes fraction', fontweight='bold', fontsize=23)

plt.savefig('fig_4.tiff', dpi=225, bbox_inches='tight')
plt.close('all')
