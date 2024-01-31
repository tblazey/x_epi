#!/usr/bin/python

#Load libs
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os


#Move into script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Various constants
plot_extent = [-128, 128, -128, 128]
epi_min = 2E-1
epi_max = 1

#Load in image headers
anat_hdr = nib.load('anat.nii.gz')
mask_hdr = nib.load('anat_mask.nii.gz')
ethy_hdr = nib.load('single_echo_eg_ud_mag_on_anat.nii.gz')
meth_hdr = nib.load('single_echo_me_ud_mag_on_anat.nii.gz')

#Load in image data
anat_data = anat_hdr.get_fdata().squeeze()
mask_data = mask_hdr.get_fdata().squeeze()
meth_data = meth_hdr.get_fdata().squeeze()
ethy_data = ethy_hdr.get_fdata().squeeze()

#Normalize image data
img_norm = np.max(np.concatenate((meth_data.flatten(), ethy_data.flatten())))
meth_data /= img_norm
ethy_data /= img_norm

#Remove zeros
anat_data[mask_data == 0] = np.nan
ethy_data[mask_data == 0] = np.nan
meth_data[mask_data == 0] = np.nan

#Remove below threshold
meth_data[meth_data < epi_min] = np.nan
ethy_data[ethy_data < epi_min] = np.nan

#Make grid figure
fig, ax = plt.subplots(1, 3, figsize=(14, 9))
plt.subplots_adjust(wspace=0, hspace=0)

#Show structural images for higher rers
ax[1].matshow(anat_data[:, :, 82].T, extent=plot_extent, 
              cmap='gray', origin='lower')
ax[2].matshow(anat_data[:, 63, ::-1].T, extent=plot_extent,
              cmap='gray', origin='lower')          
ax[0].matshow(anat_data[77, :, ::-1].T, extent=plot_extent,
              cmap='gray', origin='lower')
              
#Methanol
meth_im = ax[1].matshow(meth_data[:, :, 82].T, extent=plot_extent, 
                        cmap='plasma', origin='lower', vmin=epi_min, vmax=epi_max)
ax[2].matshow(meth_data[:, 63, ::-1].T, extent=plot_extent,
              cmap='plasma', origin='lower', vmin=epi_min, vmax=epi_max)
ax[0].matshow(meth_data[77, :, ::-1].T, extent=plot_extent,
              cmap='plasma', origin='lower', vmin=epi_min, vmax=epi_max)
              
#Ethylene glycol 
ethy_im = ax[1].matshow(ethy_data[:, :, 82].T, extent=plot_extent, 
                       cmap='viridis', origin='lower', vmin=epi_min, vmax=epi_max)
ax[2].matshow(ethy_data[:, 63, ::-1].T, extent=plot_extent,
              cmap='viridis', origin='lower', vmin=epi_min, vmax=epi_max)
ax[0].matshow(ethy_data[77, :, ::-1].T, extent=plot_extent,
              cmap='viridis', origin='lower', vmin=epi_min, vmax=epi_max)

#Turn off axes
for i in range(3):
   ax[i].axis('off')
   
#Colorbars
cax_1 = plt.axes([0.55, 0.2, 0.2, 0.03])
cb_1 = plt.colorbar(meth_im, cax=cax_1, location='bottom')
cb_1.set_label('Methanol (A.U)', weight='bold', size=11, labelpad=10)
cax_2 = plt.axes([0.3, 0.2, 0.2, 0.03])
cb_2 = plt.colorbar(ethy_im, cax=cax_2, location='bottom')
cb_2.set_label('Ethylene Glycol (A.U)', weight='bold', size=11, labelpad=10)

#Save figure
plt.savefig('fig_s3.tiff', dpi=250, bbox_inches='tight')
plt.close('all')             
