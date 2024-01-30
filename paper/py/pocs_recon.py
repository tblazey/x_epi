#!/usr/bin/python

#Load libraries
import argparse
from math import ceil
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import ifftshift, fftshift, fftn, ifftn, fftfreq

#Function to find partial fourier cutoff frequency
def compute_k_zero(pf, k_freq):
   
   #Get the index of k-space cutoff
   n_acq = ceil(pf * k_freq.shape[0])
   k_zero_idx = k_freq.shape[0] - n_acq

   #Get frequency of k-space cutoff
   k_zero = np.abs(k_freq[k_zero_idx])
      
   return k_zero, k_zero_idx

#Function that makes a low-pass filter for partial fourier recon
def homodyne_low_pass(omega, k_zero, k_freq):

   #Prep for filter creation
   k_minus = k_zero - omega / 2
   k_plus = k_zero + omega / 2
   lp_filt = np.zeros(k_freq.shape[0])
   
   #Make filter
   lp_filt[np.abs(k_freq) <= k_minus] = 1
   k_mask = np.logical_and(np.abs(k_freq) < k_plus, np.abs(k_freq) > k_minus)
   lp_filt[k_mask] = np.cos(np.pi * (np.abs(k_freq[k_mask]) - k_minus) / (2 * omega)) ** 2

   return lp_filt

#Merging filter for partial fourier recon
def homodyne_merge_filt(omega, k_zero, k_freq):

   #Prep for filter creation
   k_minus = k_zero - omega / 2
   k_plus = k_zero + omega / 2
   merge_filt = np.zeros(k_freq.shape[0])
   
   #Construct filter
   merge_filt[k_freq > -k_minus] = 1
   k_mask = np.logical_and(-k_plus < k_freq, k_freq < -k_minus)
   merge_filt[k_mask] = np.cos(np.pi * (np.abs(k_freq[k_mask]) - k_minus) / (2 * omega)) ** 2
   
   return merge_filt

#Merging function for partial fourier recon
def merge_k_space(k_acq, k_est, w_filt):
   
   return w_filt * k_acq + (1 - w_filt) * k_est

#Create parser
def get_args():
   parser = argparse.ArgumentParser(description='Iterative Partial Fourier Recon using POCS',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('img', type=str,
                       help='Nifti image containing zero-filled, complex data')
   parser.add_argument('pf', type=float, help='Partial Fourier fraction')
   parser.add_argument('out', type=str, help='Root for file outputs')
   parser.add_argument('-omega_1', type=float, default=0.1,
                       help='Weighting factor for low-pass filter.')
   parser.add_argument('-omega_2', type=float, default=0.025,
                       help='Weighting factor for merging filter')
   parser.add_argument('-pf_axis', choices=[0, 1, 2], default=2, type=int,
                       help='Undersampled axis')
   parser.add_argument('-skip_half', choices=['lower', 'upper'], default='lower',
                       help='Side of k-space that is undersampled.' \
                            ' If -epocs is specified, it is the side for first sample')
   parser.add_argument('-n_iter', type=int, default=10, help='Number of iterations')
   parser.add_argument('-plot', action='store_true', help='Output diagnostic plots')
   parser.add_argument('-update', choices=['pocs', 'lr'], default='lr',
                       help='Update method. Takes either pocs, which follows ' +
                            'Hacke et al, 1991 or lr, which only applies the low ' +
                            'resolution phase (Brown 2014, section 13.7.2)')
   parser.add_argument('-epocs', action='store_true', 
                       help='Use enhanaced POCS (Koopmans et al. 2021). Data in' \
                           ' -alt_axis must be acquired with alternate undersampling.') 
   parser.add_argument('-alt_axis', type=int, default=3, choices=[3, 4],
                       help='Axis with alternated phase encoding for epocs')
   parser.add_argument('-flip_alt', action='store_true',
                       help='Flip odd samples on alt_axis')
   return parser.parse_args()

def main():

   #Parse arguments
   args = get_args()

   #Load in image data
   img_hdr = nib.load(args.img)
   img_data = img_hdr.get_fdata(dtype=img_hdr.get_data_dtype())
   
   #Move axes so that undersampled axis is 2nd and alternated axis is 3rd
   img_data = np.moveaxis(img_data, args.pf_axis, 1)
   if args.epocs is True or args.flip_alt is True:
      img_data = np.moveaxis(img_data, args.alt_axis, 2)
   
      #Flip odd samples along alt_axis if user wants
      if args.flip_alt is True:
         img_data[:, :, 1::2] = img_data[:, ::-1, 1::2]

   #Flip undersampled axis if we are skipping the upper half
   if args.skip_half == 'upper':
      img_data = img_data[:, ::-1]

   #Get sampling frequencies (not scaled for FOV)
   n_pf_axis = img_data.shape[1]
   k_freq = fftshift(fftfreq(n_pf_axis, 1)) + 0.5 / n_pf_axis

   #Find cutoff frequency and index
   k_zero, k_zero_idx = compute_k_zero(args.pf, k_freq)
   
   #Output shape for filters
   filt_shape = np.ones(img_data.ndim, int)
   filt_shape[1] = img_data.shape[1]
   
   #Low pass filter
   lp_filt = homodyne_low_pass(args.omega_1, k_zero, k_freq).reshape(filt_shape)
   
   #Construct k-space merge filter
   m_filt = homodyne_merge_filt(args.omega_2, k_zero, k_freq)
   if args.epocs is False:
      m_filt = m_filt.reshape(filt_shape)
      
   else:
   
      #Add enhanced/alternating dimension to output shape
      n_e = img_data.shape[2]
      filt_shape[2] = n_e
      
      #Alternate every other sample in enhanced dimension
      m_filt = np.tile(m_filt[:, np.newaxis], [1, n_e])
      m_filt[:, 1::2] = m_filt[::-1, 1::2]
      m_filt = m_filt.reshape(filt_shape)

   #Get hybrid data by keeping fourier transform on fully-sampled axes
   s_data = ifftn(img_data, axes=[1])
   s_data_lp_filt = s_data * lp_filt
   
   #Get mask for sampled values
   samp_mask = np.ones(s_data.shape, dtype=bool)
   
   #Logic for enhanced vs. standard partial fourier
   if args.epocs is True:
   
      #Copy hybrid data to generate "enhanced" phase correction
      s_data_epocs = np.copy(s_data)
      
      #Removed skipped values from sampling mask
      k_zero_idx_rev = s_data.shape[1] - k_zero_idx
      samp_mask[:, 0:k_zero_idx, 0::2] = False
      samp_mask[:, k_zero_idx_rev:, 1::2] = False

      #Loop through samples in enhanced dimension
      for i in range(n_e):

         #Use next time point unless this is the last sample
         if i != n_e - 1:
            sign = 1
         else:
            sign = -1
         
         #Fill missing data with sampled data at adjacent time point
         if i % 2 == 0:
            s_data_epocs[:, 0:k_zero_idx, i] = s_data[:, 0:k_zero_idx, i + 1 * sign]
            
         else:
            s_data_epocs[:, k_zero_idx_rev::, i] = s_data[:, k_zero_idx_rev::, i + 1 * sign]
              
      #Convert back to image space
      p_phi = fftn(fftshift(s_data_epocs, axes=[1]), axes=[1])  

   else:
   
      #Compute image using symmetrically sampled region of k-space
      samp_mask[:, 0:k_zero_idx] = False
      p_phi = fftn(fftshift(s_data_lp_filt, axes=[1]), axes=[1])

   #Compute phase correction term
   phi_corr = p_phi / np.abs(p_phi)    #equal to np.exp(-1j * phi)

   #Get initial estimate of image using zero filled data
   p_hat = fftn(fftshift(s_data, axes=[1]), axes=[1])
   
   #Loop through iterations
   delta = np.zeros(args.n_iter)
   for i in range(args.n_iter):

      #Create new image, using estimated phase
      p_hat_plus = np.abs(p_hat) * phi_corr
      if args.update == 'pocs':
         p_hat_plus *= np.cos(np.angle(phi_corr * p_hat.conj()))

      #Get k-space data  of new image
      s_hat_plus = ifftshift(ifftn(p_hat_plus, axes=[1]), axes=[1])
   
      #Replace data estimate with actual data. Merge with filter if this is the last iter
      if i != args.n_iter - 1:
         s_hat_plus[samp_mask] = s_data[samp_mask]
      else:
         s_hat_plus = merge_k_space(s_data, s_hat_plus, m_filt)
      
      #Get image estimate after replacement  
      p_hat = fftn(fftshift(s_hat_plus, axes=[1]), axes=[1])
   
      #Compute difference from previous estimate
      delta[i] = np.linalg.norm(p_hat_plus.flatten() - p_hat.flatten())

   #Move axes back to normal
   if args.skip_half == 'upper':
      p_hat = p_hat[:, ::-1]
   
   if args.epocs is True or args.flip_alt is True:
      
      #Move alternating samples back
      if args.flip_alt is True:
         p_hat[:, :, 1::2] = p_hat[:, ::-1, 1::2]
      
      #Move alternating axis back
      p_hat = np.moveaxis(p_hat, 2, args.alt_axis)
   
   #Move pf axis back
   p_hat = np.moveaxis(p_hat, 1, args.pf_axis)
   
   #Save image after recon
   nib.Nifti1Image(p_hat, img_hdr.affine).to_filename(f'{args.out}.nii.gz')
   
   #Make plots if necessary
   if args.plot is True:

      #Convergence plot
      fig, ax = plt.subplots(1, 1)
      ax.grid()
      ax.plot(delta)
      ax.set_xlabel('Iteration', fontweight='bold')
      ax.set_ylabel(r'$\delta$ Error', fontweight='bold')
      ax.set_title('POCS Convergence', fontweight='bold')
      fig.savefig(f'{args.out}_conv_plot.jpeg', bbox_inches='tight')
   
      #Weighting plot
      if args.epocs is False:
         fig, ax = plt.subplots(1, 1)
         ax.grid()
         ax.plot(k_freq, lp_filt.squeeze(), label='Low Pass')
         ax.plot(k_freq, m_filt.squeeze(), label='Merge')
         ax.axvline(k_zero, ls='dashed', color='black')
         ax.axvline(-k_zero, ls='dashed', color='black')
         ax.set_xlabel('k-space', fontweight='bold')
         ax.set_ylabel('Weight', fontweight='bold')
         ax.set_title('POCS Weighting', fontweight='bold')
         ax.legend()
         fig.savefig(f'{args.out}_weight_plot.jpeg', bbox_inches='tight')
   
      plt.close('all')

if __name__ == '__main__':
    main()
