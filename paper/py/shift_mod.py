#!/usr/bin/python

#Load libs
import argparse
import nibabel as nib
import numpy as np
from numpy.fft import *
from tqdm import tqdm

#Create parser
parser = argparse.ArgumentParser(description='Apply shift map using phase modulation')
parser.add_argument('epi', help='Distorted EPI image (must be complex)')
parser.add_argument('map', help='Shift map')
parser.add_argument('out', help='Root for output image')
parser.add_argument('-axis', help='Phase encoding axis', type=int, default=1,
                    choices=[0, 1, 2])
args = parser.parse_args()

#Load in epi image
epi_hdr = nib.load(args.epi)
epi_img = epi_hdr.get_fdata(dtype=epi_hdr.get_data_dtype()).squeeze()

#Move phase encoding axis to y to make things easier
epi_swap = np.moveaxis(epi_img, args.axis, 1)

#Reshape extra dimensions if necessary
if epi_hdr.ndim > 4:
   new_dims = (epi_hdr.shape[0], epi_hdr.shape[1], epi_hdr.shape[2],
               np.prod(epi_hdr.shape[3::]))
   epi_swap = epi_swap.reshape(new_dims)
   
#Load in phase map
map_hdr = nib.load(args.map)
map_img = map_hdr.get_fdata(dtype=map_hdr.get_data_dtype()).squeeze()

#Prep for k-space modulation
epi_k = np.zeros_like(epi_swap)
mat = np.zeros((epi_img.shape[1], epi_img.shape[1]), dtype=np.complex128)
omega = np.exp(-1j * 2 * np.pi / epi_img.shape[1])

#Loop through readout dimension
for p in tqdm(range(epi_img.shape[0]), desc='Modulating'):

   #Loop through slice dimension
   for r in range(epi_img.shape[2]):
     
      #Construct matrix to apply Fourier transform + modulate with shift map
      for n in range(epi_img.shape[1]):
         for q in range(epi_img.shape[1]):
            mat[n, q] = np.power(omega, n * q + n * map_img[p, q, r])
   
      #Apply modulation
      epi_k[p, :, r] = mat @ epi_swap[p, :, r]

#Get back undistorted image 
epi_ud = np.moveaxis(ifft(epi_k, axis=1), 1, args.axis).reshape(epi_hdr.shape)

#Save undistorted image
nib.Nifti1Image(epi_ud, epi_hdr.affine).to_filename(f'{args.out}.nii.gz')




           
   