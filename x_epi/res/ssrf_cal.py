#!/usr/bin/python

# Load libraries
import numpy as np
import os
import pypulseq as pp
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
import scipy.integrate as integ
import scipy.interpolate as interp

# Set system limits
lims = pp.Opts(
    max_grad=60,
    grad_unit="mT/m",
    max_slew=160,
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    gamma=10.7084e6,
    adc_dead_time=20e-6,
    B0=3,
)
seq = pp.Sequence(system=lims)

# Constants
n_x = 512  # number of readout points
ssrf_delta = 4e-6  # temporal resolution of ssrf pulse (seconds)
delta_v = 2e3  # half the receiver bandwidth (Hz)
plot_traj = 1  # 1=plot gradient/k-space, 0=no plotting
tr = 5  # seconds
n_avg = 1  # number of averages

# Define SSRF pulses
ssrf_dir = "./"
grd_name = "siemens_singleband_pyr_3T.GRD"
rf_name = "siemens_singleband_pyr_3T.RF"
max_grd = 1.3698
grd_scale = 1
max_b1 = 0.1176
seq_out = "./ssrf_cal_fine.seq"
cal_mode = "coarse"

# Define pulse scales
if cal_mode == "fine":
    scales = np.linspace(0.75, 1.25, 12)
else:
    scale = np.arange(0.25, 3.25, 0.25)

# Load in ssrf gradient
grd_path = os.path.join(ssrf_dir, grd_name)
grd = np.loadtxt(grd_path, usecols=[0])

# Load in magnitude and phase
rf_path = os.path.join(ssrf_dir, rf_name)
pha = np.loadtxt(rf_path, usecols=[0])
mag = np.loadtxt(rf_path, usecols=[1])

# Convert gradient to Hz/m
grd *= max_grd / np.max(np.abs(grd)) * 1e2 / 1e4 * lims.gamma * grd_scale
n_grd = grd.shape[0]
t = np.arange(1, n_grd + 1) * ssrf_delta

# Convert mag and phase to Hz and radians respectively
mag *= max_b1 / np.max(mag) / 1e4 * lims.gamma
pha *= np.pi / 180

# Interpolate gradient to raster
grd_ti_end = np.ceil(t[-1] / lims.grad_raster_time) * lims.grad_raster_time
grd_ti = np.arange(
    lims.grad_raster_time, grd_ti_end + lims.grad_raster_time, lims.grad_raster_time
)

# Interpolate gradient to system raster time
grd_i = interp.interp1d(t, grd, bounds_error=False, fill_value=0.0)(grd_ti)

# Make gradient event
grd_ev = make_arbitrary_grad("z", grd_i, system=lims, delay=lims.rf_dead_time)

# Interpolate mag, phase, and kz
rf_ti = np.arange(
    lims.rf_raster_time, grd_ti_end + lims.rf_raster_time, lims.rf_raster_time
)
mag_i = interp.interp1d(t, mag, fill_value=0, bounds_error=False)(rf_ti)
pha_i = interp.interp1d(t, pha, fill_value=0, bounds_error=False)(rf_ti)

# Construct adc event
t_dwell = 1.0 / 2.0 / delta_v
t_acq = n_x * t_dwell
adc = pp.make_adc(n_x, duration=t_acq, system=lims, delay=20e-6)

# Loop through frequencies
for scale in scales:
    # Make rf event
    rf_ev = pp.make_arbitrary_rf(
        scale * mag_i * np.exp(1j * pha_i), 2 * np.pi, system=lims, norm=False
    )
    rf_dur = rf_ev.dead_time + rf_ev.ringdown_time + rf_ev.shape_dur
    delay_aug = (
        np.ceil(rf_dur / lims.block_duration_raster) * lims.block_duration_raster
    )
    if delay_aug > 0:
        rf_delay = pp.make_delay(delay_aug)

    # Loop through averages
    for avg in np.arange(n_avg):
        # Excitation block
        block_start = len(seq.block_durations)
        if delay_aug > 0:
            seq.add_block(rf_ev, rf_delay, grd_ev)
        else:
            seq.add_block(rf_ev, grd_ev)

        # Readout block
        seq.add_block(adc)

        # Relaxation delay
        tr_delay = tr - np.sum(seq.block_durations[block_start::])
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

# Save sequence
seq.write(seq_out)

# Plot sequence if necessary
if plot_traj == 1:
    seq.plot()

print("Total sequence duration: %f (s)", np.sum(seq.block_durations))
