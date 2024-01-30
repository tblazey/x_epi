#!/usr/bin/python

#Load libs
import argparse
import nibabel as nib
import numpy as np

#Create parser
parser = argparse.ArgumentParser(description='Compute SNR for multi-echo image')
parser.add_argument('img', help='Input image')
parser.add_argument('out', help='Root for output image')
parser.add_argument('-x_range', default=[0, 4], nargs=2,
                    help='Indices for background roi in x dimension')
parser.add_argument('-y_range', default=[0, 4], nargs=2,
                    help='Indices for background roi in y dimension')
parser.add_argument('-z_range', default=[0, 4], nargs=2,
                    help='Indices for background roi in z dimension')
parser.add_argument('-scale', default=0.665, type=float, 
                    help='Scale factor to adjust for Rayleigh distribution')                                        
args = parser.parse_args()

#Load in multi-echo image
hdr = nib.load(args.img)
img = hdr.get_fdata()

#Compute snr using background region
roi_std = np.std(img[args.x_range[0]:args.x_range[1],
                     args.y_range[0]:args.y_range[1],
                     args.z_range[0]:args.z_range[1]], axis=(0, 1, 2))
snr = img / roi_std * args.scale

#Write out images
nib.Nifti1Image(snr, hdr.affine).to_filename(f'{args.out}.nii.gz')
