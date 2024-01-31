#!/usr/bin/python

#Load libs
import argparse
import nibabel as nib
import numpy as np

#Create parser
parser = argparse.ArgumentParser(description='Swap dimensions',
                                 epilog='Note: Changes image data but not sform/qform')
parser.add_argument('img', help='Input image')
parser.add_argument('out', help='Root for output image')
parser.add_argument('axes', help='New order of axes', nargs='+', type=int)
parser.add_argument('-add', type=int,
                    help='Add additional dimensions at end before reshapping')
args = parser.parse_args()

#Load in image data
img_hdr = nib.load(args.img)
img_data = img_hdr.get_fdata(dtype=img_hdr.get_data_dtype())

#Add additional dimensions
if args.add is not None:
    r_dims = np.concatenate((img_hdr.shape, np.repeat(1, args.add)))
    img_data = img_data.reshape(r_dims)

#Write out squeezed image
img_out = nib.Nifti1Image(np.transpose(img_data, args.axes), img_hdr.affine)
img_out.to_filename(f'{args.out}.nii.gz')
