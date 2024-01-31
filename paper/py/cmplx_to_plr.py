#!/usr/bin/python

#Load libs
import argparse
import nibabel as nib
import numpy as np

#Create parser
parser = argparse.ArgumentParser(description='Converts complex image to polar')
parser.add_argument('img', help='Complex image')
parser.add_argument('out', help='Root for output image')
args = parser.parse_args()

#Load in image
hdr = nib.load(args.img)
img = hdr.get_fdata(dtype=hdr.get_data_dtype())

#Write out magnitude
mag = nib.Nifti1Image(np.abs(img), hdr.affine)
mag.to_filename(f'{args.out}_mag.nii.gz')

#Write out phase
pha = nib.Nifti1Image(np.angle(img), hdr.affine)
pha.to_filename(f'{args.out}_pha.nii.gz')
