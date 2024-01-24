"""
Computes power scale factor from calibration sequence
"""

# Load libs
import argparse
import matplotlib.pyplot as plt
from nmrglue import proc_autophase
import numpy as np
from numpy.fft import fftn, fftshift
from scipy.optimize import minimize
from twixtools import read_twix


def create_parser():
    """
    Creates argparse argument parser for calibration script
    """

    parser = argparse.ArgumentParser(
        description="Estimates power for x_epi sequences on Siemens scanners"
    )
    parser.add_argument("twix", help="Siemsn Twix file containing EPI data")
    parser.add_argument("-n_avg", default=1, help="Number of averages")
    parser.add_argument(
        "-angle",
        default=90,
        type=float,
        help="Flip angle to return caclibration sale for. Default is 90 degrees.",
    )
    parser.add_argument(
        "-mode",
        choices=["coarse", "fine"],
        default="coarse",
        help="Calibration sequence run. Default is coarse",
    )

    return parser


def load_data(twix_path, n_avg=1):
    """
    Loads in calibration data in twix format

    Parameters
    ----------
    twix_path : str
        Path to twix file containing calibration data
    n_avg : int
        Number of averages performed during calibration

    Returns
    -------
    fid_data : ndarray
        Numpy array containing calibration spectra. Will have dimensions points x scales.
    """

    # Get twix data
    twix_data = read_twix(twix_path)
    n_data = len(twix_data[-1]["mdb"])
    fid_data = np.array([twix_data[-1]["mdb"][i].data for i in range(n_data)]).squeeze()

    # Sum averages if necessary
    fid_data = fid_data.reshape((fid_data.shape[0] // n_avg, n_avg, fid_data.shape[1]))
    fid_data = np.sum(fid_data, axis=1)

    return fid_data


def recon_spec(fid_data):
    """
    Computes FFT and phase corrects individual spectra

    Parameters
    ----------
    fid_data : ndarray
        Array containing FID with shape points x scales.

    Returns
    -------
    pc_data : ndarray
        Array containing phase-corrected spectra with shape points x scales
    """

    # Run fft on each spectra
    spec_data = fftshift(fftn(fid_data, axes=[1]), axes=[1])

    # Use spectra with highest magnitude to compute phase correction angles
    max_idx = np.unravel_index(np.argmax(np.abs(spec_data[0:7, :])), spec_data.shape)[0]
    _, phases = proc_autophase.autops(spec_data[max_idx, :], "acme", return_phases=True)

    # Apply phase correction
    pc_data = np.zeros_like(spec_data)
    for i in range(spec_data.shape[0]):
        pc_data[i, :] = proc_autophase.ps(spec_data[i, :], p0=phases[0], p1=phases[1])
    pc_data /= np.max(np.abs(pc_data))

    return pc_data


def sin_model(x, t):
    """
    Sinusoidal model for power calibration

    Parameters
    ----------
    x : ndarray
        Parameters for sinusoidal model (amplitude, phase, offset)
    t : ndarray
        Sampling points

    Returns
    -------
    hat : ndarray
        Model estimates at `t` given parameters in `x`
    """

    return x[0] * np.sin(x[1] * t + x[2])


def sin_cost(x, t, y):
    """
    Cost function for sinusoidal model

    Parameters
    ----------
    x : ndarray
        Parameters for sinusoidal model (amplitude, phase, offset)
    t : ndarray
        Sampling points
    y : ndarray
        Measured data at sampling points `t`

    Returns
    -------
    cost : float
        Sum of squares between model estimate and y
    """

    return np.sum(np.power(y - sin_model(x, t), 2))


def fit_cal_data(cal_data, scales):
    """
    Run sinusoidal fitting on calibration data

    Parameters
    ----------
    cal_data : ndarray
        Signal for each power level in `scales`
    scales : ndarray
        Power calibration scale factors

    Returns
    -------
    x_hat : ndarray
        Estimated parameters for sinusoidal model (amplitude, phase, offset)
    """

    fit = minimize(sin_cost, [1, 0, 0], args=(scales, cal_data))
    return fit.x


def main(argv=None):
    """
    Run calibration script
    """

    # Run parser
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load in twix data
    fid_data = load_data(args.twix, n_avg=args.n_avg)

    # Recon + Phase correction
    spec_data = recon_spec(fid_data)

    # Get signal around peak for each power scale
    max_idx = np.unravel_index(np.argmax(np.real(spec_data)), spec_data.shape)[1]
    cal_data = np.sum(np.real(spec_data[:, max_idx - 4 : max_idx + 5]), axis=1)
    cal_data /= np.max(np.abs(cal_data))

    if args.mode == "coarse":
        scales = np.arange(0.25, 3.25, 0.25)
    else:
        scales = np.linspace(0.75, 1.25, 12)

    # Run model fitting
    par_hat = fit_cal_data(cal_data, scales)

    # Make figure with spectra
    plt.figure(0)
    plt.grid()
    for i in range(spec_data.shape[0]):
        plot_x = np.arange(0, spec_data.shape[1]) + i * spec_data.shape[1]
        plt.plot(plot_x, np.real(spec_data[i, :]))
    plt.ylabel("Normalized Signal", fontweight="bold")
    plt.title("Calibration Specta", fontweight="bold")

    # Get scale for requested flip angle
    scale_hat = (np.pi / 2 - par_hat[2]) / par_hat[1]
    scale_cal = args.angle * scale_hat / 90

    # Make model estimate figure
    x_i = np.linspace(scales[0], scales[-1], 200)
    y_i = sin_model(par_hat, x_i)
    plt.figure(1)
    plt.grid()
    plt.scatter(scales, cal_data, marker="o", c="black", zorder=2)
    plt.plot(x_i, y_i, c="#006bb6", zorder=1, lw=2.5)
    plt.xlabel("Scale", fontweight="bold")
    plt.ylabel("Intensity", fontweight="bold")
    plt.title(f"Estimated Scale = {scale_cal:0.3f}", fontweight="bold")

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
