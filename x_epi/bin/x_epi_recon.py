"""
Command line script for reconstruction of images acquired with x_epi
"""

# Load libs
import argparse
import json
from fsl.data.image import Image
from fsl.transform.flirt import sformToFlirtMatrix
import numpy as np
from x_epi.data import XData


def create_parser():
    """
    Creates argparse argument parser for recon script
    """

    # Create parser
    parser = argparse.ArgumentParser(
        description="Reconstruction for data acquired with x_epi"
    )
    parser.add_argument("twix", help="Twix file containing EPI data")
    parser.add_argument("json", help="Json file describing sequence from x_epi_app.py")
    parser.add_argument("out", help="Root for file outputs")
    parser.add_argument("-anat", help="Anatomical reference image")
    parser.add_argument(
        "-complex", action="store_true", help="Save complex image as well as magnitude."
    )
    parser.add_argument(
        "-freq_off",
        nargs="+",
        type=float,
        help="Additional off-resonance for each metabolite",
    )
    parser.add_argument(
        "-k_coord", type=str, help="k-space coordinates for ramp sampling recon"
    )
    parser.add_argument("-mean", help="Output temporal means", action="store_true")
    parser.add_argument(
        "-n_k",
        default=6,
        type=int,
        metavar="KNOTS",
        help="Number of knots for spline. Default is 6. "
        "Using 2 knots will give linear fit",
    )
    parser.add_argument("-n_avg", type=int, help="Number of total averages", default=1)
    parser.add_argument(
        "-n_rep", type=int, help="Number of total repetitions", default=1
    )
    parser.add_argument(
        "-point",
        action="store_true",
        help="Run pointwise phaes correction. Default is spline based.",
    )
    parser.add_argument(
        "-ref_info", help="Twix data and json file for reference scan", nargs=2
    )
    parser.add_argument("-ts", type=float, help="Sampling time (s)")
    parser.add_argument("-save_k", action='store_true', help="Saves k-space data")

    return parser


def extract_pars(json_path, n_avg=None, n_rep=None, ts=None, freq_off=None):
    """
    Extract image parameters for x_epi json file

    Parameters
    ----------
    json_path : str
        Path to parameter json file
    n_avg : int
        Number of averages. Overides value in json file.
    n_rep : int
        Number of repetitions. Overrides value in json file.
    ts : int
        Temporal sampling time. Overrides value in json file.
    freq_off : float
        Additional off-resonance term (Hz) added to each metabolite.

    Returns
    -------
    x_data : XData object
        Object containing sequence parameters

    Notes
    -----
    The n_avg, n_rep, and ts options are added because these parameters can be changed
    with the Siemens Pulseq interpreter sequence.
    """

    # Read in json file containing sequence pars
    with open(json_path, "r", encoding="utf-8") as jid:
        param_dic = json.load(jid)

    # Update temporal parameters
    if n_avg is not None:
        param_dic["n_avg"] = n_avg
    if n_rep is not None:
        param_dic["n_rep"] = n_rep
    if ts is not None:
        param_dic["ts"] = ts

    # Add general parameters
    seq_dic = {key: param_dic[key] for key in param_dic if key != "mets"}
    x_data = XData(**seq_dic)

    # Add metabolites
    met_list = param_dic["mets"]
    for idx, met in enumerate(met_list):
        if freq_off is not None:
            met["freq_off"] += freq_off[idx]
        x_data.add_met(**met)

    return x_data


def run_preproc(
    twix_path, json_path, n_avg=None, n_rep=None, ts=None, freq_off=None, k_coord=None
):
    """
    Runs preprocessing of x_epi data

    Parameters
    ----------
    twix_path : str
        Path to Siemens twix data
    json_path : str
        Path to x_epi parameter file in JSON format
    n_avg : int
        Number of averages
    n_rep : int
        Number of repetitions
    ts : float
        Sampling time
    freq_off : list
        Additional frequency offset for each metabolite
    k_coord : str
        Path to k-space coordinates from x_epi apps

    Returns
    -------
    x_data : XData object
        Data object containing image data and parameters
    """

    # Create XData object from json parameter file
    x_data = extract_pars(json_path, n_avg=n_avg, n_rep=n_rep, ts=ts, freq_off=freq_off)

    # Get k-space data from twix file
    x_data.load_k_data(twix_path, recon_dims=k_coord is None)

    # Regrid if necessary
    if k_coord is not None:
        x_data.load_k_coords(k_coord)
        x_data.regrid_k_data(method='nufft')

    # Flip axes
    x_data.flip_k_data()

    return x_data


def save_xfms(anat_path, met_list, out_root):
    """
    Creates transformation between reconstructed images and a anatomical reference

    Parameters
    ----------
    anat_path : str
        Path to anatomical reference Nifti image
    met_list : list
        List containing metabolite objects created by XData
    out_root : str
        Root for all output files (no extensions)
    """

    # Load in anatomical image
    anat_img = Image(anat_path)

    # Loop through each metabolite
    for met in met_list:
        # Load EPIimage that we just wrote, not very efficient
        img = Image(out_root + f"_{met.name}")

        # Compute and save transformation between EPI and anatomical reference
        xfm = sformToFlirtMatrix(img, anat_img)
        xfm_path = out_root + f"_{met.name}_to_{anat_img.name}.mat"
        np.savetxt(xfm_path, xfm)


def main(argv=None):
    """
    Run recon script
    """

    # Run parser
    parser = create_parser()
    args = parser.parse_args(argv)

    # Get k-space data and parameters for scan
    run_keys = ["n_avg", "n_rep", "ts", "freq_off", "k_coord"]
    kwargs = {key: value for key, value in vars(args).items() if key in run_keys}
    x_data = run_preproc(args.twix, args.json, **kwargs)

    # Get k-space data and pars for reference scan
    if args.ref_info is not None:
        x_ref = run_preproc(args.ref_info[0], args.ref_info[1], k_coord=args.k_coord)
    else:
        x_ref = None

    # Reconstruct images, applying phase correction if reference scan is supplied
    x_data.fft_recon(ref_data=x_ref, slice_idx=None, n_k=args.n_k, point=args.point)

    # Shift off resonance metabolites
    x_data.apply_off_res()

    # Phase shift(s) for off-center FOV
    x_data.apply_phase_shift()

    # Save images
    x_data.save_nii(args.out)
    if args.complex is True:
        x_data.save_nii(args.out, cmplx=True)
    if args.mean is True:
        x_data.save_nii(args.out, mean=True)

    # Save transformations to an anatomical reference image
    if args.anat is not None:
        save_xfms(args.anat, x_data.mets, args.out)

    # Save metabolite parameters to json
    x_data.save_param_dic(args.out)
    
    # Save k-space data
    if args.save_k is True:
        for met in x_data.mets:
            if met.k_acq is not None:
                np.save(f'{args.out}_{met.name}_k_acq.npy', met.k_acq)
            np.save(f'{args.out}_{met.name}_k_data.npy', met.k_data)
            

if __name__ == "__main__":
    main()
