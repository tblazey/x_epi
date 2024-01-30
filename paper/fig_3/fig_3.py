#!/usr/bin/python

#Load libs
from itertools import product
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import scipy.optimize as opt

#Various constants
plot_extent = [-128, 128, -128, 128]
epi_min = 0.2
epi_max = 1
eg_snr_min = 7.5
eg_snr_max = 25
me_snr_min = 5
me_snr_max = 15
n_echo = 5 
delta_te = np.array([0.01968, 0.01056])[::-1]
right_col = '#b6006b'
left_col = '#006bb6'
eg_map = 'viridis'
met_map = 'plasma'

#Load in image headers
anat_hdr = nib.load('../fig_2/anat.nii.gz')
mask_hdr = nib.load('../fig_2/anat_mask.nii.gz')
ethy_hdr = nib.load('multi_echo_eg_ud_mag_on_anat.nii.gz')
meth_hdr = nib.load('multi_echo_me_ud_mag_on_anat.nii.gz')
c1_hdr = nib.load('c1_on_anat.nii.gz')
c2_hdr = nib.load('c2_on_anat.nii.gz')

#Load in image data
anat_data = anat_hdr.get_fdata().squeeze()
mask_data = mask_hdr.get_fdata().squeeze()
meth_data = meth_hdr.get_fdata().squeeze()
ethy_data = ethy_hdr.get_fdata().squeeze()
c1_data = c1_hdr.get_fdata().squeeze()
c1_mask = c1_data.flatten() == 1
c2_data = c2_hdr.get_fdata().squeeze()
c2_mask = c2_data.flatten() == 1

#Loop through images/echo means
snr_list = []
for met, img in product(['eg', 'me'], ['ud', 'ud_echo_mean']):
   img_data = nib.load(f'multi_echo_{met}_{img}_snr_on_anat.nii.gz').get_fdata().squeeze()
   if img_data.ndim == 4:
      img_data = img_data[:, :, :, 0]
   snr_list.append(img_data)
   if met == 'eg':
      snr_mean = np.mean(img_data.flatten()[c2_mask])
   else:
      snr_mean = np.mean(img_data.flatten()[c1_mask])
   print(f'{met}, {img}, {snr_mean}')

#Normalize image data
img_norm = np.max(np.concatenate((meth_data.flatten(), ethy_data.flatten())))
meth_data /= img_norm
ethy_data /= img_norm

#Get means for each echo within each compartment (met, comp, echo)
echo_means = np.zeros((2, 2, n_echo))
meth_2d = meth_data.reshape((np.prod(meth_data.shape[0:3]), meth_data.shape[3]))
ethy_2d = ethy_data.reshape((np.prod(ethy_data.shape[0:3]), ethy_data.shape[3]))
echo_means[0, 0, :] = np.mean(meth_2d[c1_mask, :], axis=0)
echo_means[1, 0, :] = np.mean(ethy_2d[c1_mask, :], axis=0)
echo_means[0, 1, :] = np.mean(meth_2d[c2_mask, :], axis=0)
echo_means[1, 1, :] = np.mean(ethy_2d[c2_mask, :], axis=0)
echo_means /= np.max(echo_means)

#Exponential decay model
def t2_model(params, t):
   return params[0] * np.exp(-t / params[1])
   
#Cost function
def t2_fit(params, t, y):
   return np.sum(np.power(y - t2_model(params, t), 2)) * 0.5
   
#Jacobian
def t2_jac(params, t, y):
   resid = y - t2_model(params, t)
   d_1 = np.sum(-resid * np.exp(-t / params[1]))
   d_2 = np.sum(-resid * params[0] * t * np.exp(-t / params[1]) / params[1]**2 )
   return np.array([d_1, d_2])

#Hessian  
def t2_hess(params, t, y):
   d_11 = np.sum(np.exp(-2 * t / params[1]))
   d_12 = np.sum(t * (2 * params[0] * np.exp(-t / params[1]) - y) * np.exp(-t / params[1]) / params[1]**2)
   d_21 = d_12
   d_22 = np.sum(params[0] * t * (params[0] * t * np.exp(-t / params[1]) / params[1] - \
                               2 * params[0] * np.exp(-t / params[1]) + \
                               t * (params[0] * np.exp(-t / params[1]) - y) / params[1] + \
                               2 * y) * np.exp(-t / params[1]) / params[1] ** 3)
   return np.array([[d_11, d_12], [d_21, d_22]]) 

#Run fit
te = np.arange(n_echo)[np.newaxis, :] * delta_te[:, np.newaxis]   
t2_pars = np.zeros((2, 2, 4))
for i in range(2):
   for j in range(2):
      fit = opt.minimize(t2_fit, [echo_means[i, j, 0], 50E-3], jac=t2_jac, hess=t2_hess,
                         args=(te[i, :], echo_means[i, j, :]), method='Newton-CG')
      t2_pars[i, j, 0:2] = fit.x
      hess = t2_hess(fit.x, te[i, :], echo_means[i, j, :])
      sigma_sq = t2_fit(fit.x,  te[i, :], echo_means[i, j, :]) / (te.shape[1] - 2)
      cov = sigma_sq * np.linalg.inv(hess)
      t2_pars[i, j, 2::] = np.sqrt(np.diag(cov)) #standard errors/standard deviation

#Remove zeros
anat_data[mask_data == 0] = np.nan

#Remove below threshold
meth_data[meth_data < epi_min] = np.nan
ethy_data[ethy_data < epi_min] = np.nan

#Make gridspec figure
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 5, height_ratios=[0.7, 0.2, 1])

#Make line plot showing signal in each compartment
lax = fig.add_axes([0.525, 0.11, 0.375, 0.35])#fig.add_subplot(gs[2, 3::])
lax.grid()
lax.plot(te[0, :] * 1E3, echo_means[0, 0, :], ls='dashed', color=right_col, zorder=0)  #met,  c1
lax.plot(te[1, :] * 1E3, echo_means[1, 0, :], ls='solid', color=right_col, zorder=0)   #eg, c1 
lax.plot(te[0, :] * 1E3, echo_means[0, 1, :], ls='dashed', color=left_col, zorder=0)   #met, c2
lax.plot(te[1, :] * 1E3, echo_means[1, 1, :], ls='solid', color=left_col, zorder=0)    #eg,  c2
lax.scatter(te[0, :] * 1E3, echo_means[0, 0, :], color='black', zorder=1)
lax.scatter(te[1, :] * 1E3, echo_means[1, 0, :], color='black', zorder=1)
lax.scatter(te[0, :] * 1E3, echo_means[0, 1, :], color='black', zorder=1)
lax.scatter(te[1, :] * 1E3, echo_means[1, 1, :], color='black', zorder=1)
lax.set_xlabel('Rel. Echo Time (ms)', fontweight='bold', fontsize=10)
lax.set_ylabel('Intensity (A.U.)', fontweight='bold', fontsize=10)
lax.tick_params(axis='both', labelsize=8)
lax_handles = [lines.Line2D([], [], color='black'),
               lines.Line2D([], [], color='black', ls='dashed')]
leg = lax.legend(handles=lax_handles, labels=[r'(CH$_2$OH)$_2$', r'CH$_3$OH'],
           ncols=2, loc='upper left', bbox_to_anchor=(0.25, 0.975),
           title='Metabolite Img.', columnspacing=0.2,
           fontsize=8, title_fontproperties={'size':8, 'weight':'bold'}, frameon=False)
leg._legend_box.sep = 10

#Add T2* estimates
met_t2_hat = np.int32(np.round(t2_pars[0, 0, 1] * 1E3))
met_t2_err = np.int32(np.round(t2_pars[0, 0, 3] * 1E3 * 1.96))
eg_t2_hat = np.int32(np.round(t2_pars[1, 1, 1] * 1E3))
eg_t2_err = np.int32(np.round(t2_pars[1, 1, 3] * 1E3 * 1.96))
met_str = r'$\bf{T_2^*}$' + f'= {met_t2_hat} ms [{met_t2_hat - met_t2_err}, {met_t2_hat + met_t2_err}]'
eg_str = r'$\bf{T_2^*}$' + f'= {eg_t2_hat} ms [{eg_t2_hat - eg_t2_err}, {eg_t2_hat + eg_t2_err}]'
lax.annotate(met_str,  [0.15, 0.2], fontsize=9, fontweight='bold',
             xycoords='axes fraction', color=right_col)
lax.annotate(eg_str, [0.325, 0.625], fontsize=9, fontweight='bold',
             xycoords='axes fraction', color=left_col)

#Add roi image
ax_roi = fig.add_axes([0.75, 0.274, 0.15, 0.15], zorder=1)
ax_roi.axis('off')
ax_roi.matshow(anat_data[:, :, 75].T, extent=plot_extent, cmap='gray', origin='lower')
ax_roi.matshow(c1_data[:, :, 75].T, extent=plot_extent, cmap='Blues', vmin=0.5, vmax=0.9,
               origin='lower')
ax_roi.matshow(c2_data[:, :, 75].T, extent=plot_extent, cmap='Reds', vmin=0.5, vmax=0.9,
               origin='lower')
ax_roi.images[1].cmap.set_over(right_col) #c1
ax_roi.images[2].cmap.set_over(left_col)  #c2
for img in ax_roi.images:
   img.cmap.set_under(alpha=0)
ax_roi.set_title('Compartment', y=0.975, fontweight='bold', fontsize=8)

#Show images
for i, j in enumerate(range(0, n_echo)):
   ax = fig.add_subplot(gs[0, i])
   ax.matshow(anat_data[:, :, 75].T, extent=plot_extent,
                 cmap='gray', origin='lower')          
   meth_im = ax.matshow(meth_data[:, :, 75, j].T, extent=plot_extent, 
                           cmap=met_map, origin='lower', vmin=epi_min, vmax=epi_max)
   ethy_im = ax.matshow(ethy_data[:, :, 75, j].T, extent=plot_extent, 
                           cmap=eg_map, origin='lower', vmin=epi_min, vmax=epi_max)
   ax.axis('off')
   
   #Add echo label
   #if i == 1:
   ax.set_title(f'Echo {j + 1}', fontweight='bold', y=1.01, fontsize=14)
   
#Add single echo SNR
ax_snr = fig.add_subplot(gs[2, 0])
ax_snr.matshow(anat_data[:, :, 75].T, extent=plot_extent, cmap='gray', origin='lower')
meth_snr = ax_snr.matshow(snr_list[2][:, :, 75].T, extent=plot_extent, 
                         cmap=met_map, origin='lower', vmin=me_snr_min, vmax=me_snr_max)
ethy_snr = ax_snr.matshow(snr_list[0][:, :, 75].T, extent=plot_extent, 
                          cmap=eg_map, origin='lower', vmin=eg_snr_min, vmax=eg_snr_max) 
ax_snr.set_title('Single Echo', fontweight='bold', fontsize=14, y=1.01)
ax_snr.axis('off')                          
for img in ax_snr.images:
   img.cmap.set_under(alpha=0)   
   
#Add mean over echo SNR
ax_mean = fig.add_subplot(gs[2, 1])
ax_mean.matshow(anat_data[:, :, 75].T, extent=plot_extent, cmap='gray', origin='lower')
ax_mean.matshow(snr_list[3][:, :, 75].T, extent=plot_extent, 
               cmap=met_map, origin='lower', vmin=me_snr_min, vmax=me_snr_max)
ax_mean.matshow(snr_list[1][:, :, 75].T, extent=plot_extent, 
                cmap=eg_map, origin='lower', vmin=eg_snr_min, vmax=eg_snr_max)
ax_mean.set_title('Mean', fontweight='bold', fontsize=14, y=1.01)                 
ax_mean.axis('off')   
for img in ax_mean.images:
   img.cmap.set_under(alpha=0)                               

#Colorbars for individual echoes
cax_1 = plt.axes([0.55, 0.575, 0.2, 0.02])
cb_1 = plt.colorbar(meth_im, cax=cax_1, location='bottom')
cb_1.set_label('Methanol (A.U)', weight='bold', size=8, labelpad=4)
cb_1.ax.tick_params(labelsize=7)
cax_2 = plt.axes([0.3, 0.575, 0.2, 0.02])
cb_2 = plt.colorbar(ethy_im, cax=cax_2, location='bottom')
cb_2.set_label('Ethylene Glycol (A.U)', weight='bold', size=8, labelpad=4)
cb_2.ax.tick_params(labelsize=7)

#Colorbars for SNR
cax_3 = plt.axes([0.31, 0.125, 0.1, 0.02])
cb_3 = plt.colorbar(meth_snr, cax=cax_3, location='bottom')
cb_3.set_label('Methanol (SNR)', weight='bold', size=8, labelpad=4)
cb_3.ax.tick_params(labelsize=7)
cax_4 = plt.axes([0.16, 0.125, 0.1, 0.02])
cb_4 = plt.colorbar(ethy_snr, cax=cax_4, location='bottom')
cb_4.set_label('Ethylene Glycol (SNR)', weight='bold', size=8, labelpad=4)
cb_4.ax.tick_params(labelsize=7)

#Subfig annotations
ax.annotate('A)', (0.05, 0.82), xycoords='figure fraction', fontweight='bold', 
            fontsize=20)
ax.annotate('B)', (0.05, 0.4), xycoords='figure fraction', fontweight='bold', 
            fontsize=20)
ax.annotate('C)', (0.4, 0.4), xycoords='figure fraction', fontweight='bold', 
            fontsize=20)

#Save figure
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.savefig('fig_3.tiff', dpi=250, bbox_inches='tight')

plt.close('all')             




      

