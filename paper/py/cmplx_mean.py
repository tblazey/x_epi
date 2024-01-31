#!/usr/bin/python

#Load libs
import argparse
import nibabel as nib
import numpy as np

#Create parser
parser = argparse.ArgumentParser(description='Average complex image')
parser.add_argument('img', help='Multi-echo image')
parser.add_argument('out', help='Root for output image')
parser.add_argument('axes', help='Axes to compute mean along (starts at zero)',
                    type=int, nargs='+')
parser.add_argument('-mag', help='Return magnitude image', action='store_true')
parser.add_argument('-sum', help='Compute sum instead of mean', action='store_true')
args = parser.parse_args()

#Load in image
hdr = nib.load(args.img)
img = hdr.get_fdata(dtype=hdr.get_data_dtype())

#Compute mean
if args.sum is True:
    mean = np.sum(img, axis=tuple(args.axes))
else:
    mean = np.mean(img, axis=tuple(args.axes))

#Compute magnitude if necessary
if args.mag is True:
   mean = np.abs(mean)

#Save
nib.Nifti1Image(mean, hdr.affine).to_filename(f'{args.out}.nii.gz')

