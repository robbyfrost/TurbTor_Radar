# ------------------------------------------------
# Name: functions.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 10 September 2024
# Purpose: Functions for using lidar data
# ------------------------------------------------

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import xarray as xr
import xrft
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
import scipy.ndimage as ndimage

# ------------------------------------------------
# functions
def north0_to_arctheta(theta):
    """
    Convert from mathematical degrees to meteorological (zero degrees is north).

    Parameters:
    - theta: Array of azimuths
    """
    return  np.where(theta > 270, 450-theta, 90-theta)

def beam_height_2D(r, ele, instrument_height=0.0):
    """
    Calculate lidar beam height

    Parameters:
    - r: Array of ranges [m]
    - ele: Elevation angle of lidar scan
    - intrument_height: Altitude of lidar when scan occured
    """
    a = 6.371e3 # Earth's Radius
    ae = 4 / 3 * a # Effective Earth's radius
    bh = np.sqrt((r[:,np.newaxis])**2 + ae**2 + (2 * ae * r)[:,np.newaxis].dot(np.sin(ele * np.pi / 180)[np.newaxis,:])) - ae + instrument_height
    return bh

def beam_range_2D(r_km,ele, instrumentheight=0.0):
    """
    Calculate lidar beam range

    Parameters:
    - r: Array of ranges [m]
    - ele: Elevation angle of lidar scan
    - intrument_height: Altitude of lidar when scan occured
    """
    a = 6.371e3 # Earth's Radius
    ae = 4 / 3 * a # Effective Earth's radius
    h = beam_height_2D(r_km, ele, instrumentheight) # Beam Height
    br = ae * np.arcsin(r_km[:,np.newaxis].dot(np.cos(ele * np.pi / 180)[np.newaxis,:]) / (ae + h))
    return br,h

def dis_angle_to_2Dxy(r,theta):
    """
    Calculate cartesian grid from lidar object

    Parameters:
    - r: Array of ranges
    - theta: Array of azimuths
    """
    X = r[:,np.newaxis].dot(np.cos(theta[np.newaxis,:] * np.pi / 180))
    Y = r[:,np.newaxis].dot(np.sin(theta[np.newaxis,:] * np.pi / 180))
    return X,Y

def inpaint_nans(data, mask, sigma=3):
    """
    Inpaint NaNs in data using a Gaussian filter to create a smoother background.
    
    Parameters:
    - data: 2D array with NaNs where data is masked out
    - mask: Boolean array, True where data should be masked
    - sigma: Smoothing parameter for Gaussian filter
    
    Returns:
    - inpainted data: 2D array with NaNs replaced by smoothed values
    """
    # Replace NaNs with zero and smooth over non-masked regions
    filled_data = np.where(mask, data, 0)
    smooth_data = gaussian_filter(filled_data, sigma=sigma)
    
    # Ensure original data values remain where mask is False
    inpainted_data = np.where(mask, smooth_data, data)
    
    return inpainted_data

def extrapolate_nans(data, mask):
    # Get the indices of valid and NaN (masked) points
    valid_points = np.where(~mask)
    nan_points = np.where(mask)
    
    # Perform a nearest-neighbor interpolation to fill NaNs and extrapolate
    filled_data = ndimage.map_coordinates(
        data, 
        np.array([valid_points[0], valid_points[1]]), 
        order=1, 
        mode='nearest'
    )

    # Replace the masked areas with interpolated/extrapolated values
    data[mask] = data[mask] = filled_data[:np.sum(mask)]
    return data

def find_phase_shift(original, filtered):
    # Flatten the data along the azimuthal dimension
    original_flat = original.mean(dim='r').values
    filtered_flat = filtered.mean(dim='r').values

    # Compute cross-correlation
    correlation = correlate(original_flat, filtered_flat, mode='full')
    shift_index = correlation.argmax() - (len(correlation) // 2)

    # Convert the shift index to an angle
    angle_shift = (shift_index / original.az.size) * 360  # Convert to degrees
    return angle_shift

def low_pass_filter(var, az, r, cutoff, snr_mask):
    """
    Apply a low-pass filter to a 2D (azimuth, range) lidar array

    Parameters:
    - var: Lidar variable to apply filter to with dimensions (azimuth, range)
    - az: Array of azimuths corresponding to var
    - r: Array of ranges corresponding to var
    - cutoff: (Float) cutoff wavenumber
    - snr_mask: Masked array where snr > some cutoff value
    """
    # grab data
    v = xr.DataArray(var, dims=("az","r"), coords={'az': az, 'r': r}).fillna(0)
    # Create uniform coordinates for azimuth and range
    az_uniform = np.linspace(v.az.min().data, v.az.max().data, v.shape[0])
    r_uniform = np.linspace(v.r.min().data, v.r.max().data, v.shape[1])
    # Interpolate data onto uniform grid
    var_uniform = v.interp(az=az_uniform, r=r_uniform, method='linear')
    # # use Gaussian filter for beam blockage
    # var_filled = inpaint_nans(var_uniform.values, ~snr_mask)
    var_filled = extrapolate_nans(var_uniform.values, ~snr_mask)
    var_uniform_filled = xr.DataArray(var_filled, dims=("az", "r"), coords=var_uniform.coords)
    # take fft
    f_var = xrft.fft(var_uniform_filled, dim=('az','r'), true_phase=True, true_amplitude=True)
    
    # take cutoff wavenumber such as 1/5
    fc = cutoff
    # zero out x and y wavenumbers above this cutoff to get lowpass filter
    jrp = np.where(f_var.freq_r > fc)[0]
    jrn = np.where(f_var.freq_r < -fc)[0]
    jazp = np.where(f_var.freq_az > fc)[0]
    jazn = np.where(f_var.freq_az < -fc)[0]
    f_var[:,jrp] = 0
    f_var[:,jrn] = 0
    f_var[jazp,:] = 0
    f_var[jazn,:] = 0

    # take ifft
    var_filt = xrft.ifft(f_var, dim=('freq_az','freq_r'),
                        true_amplitude=True, true_phase=True,
                        lag=(f_var.freq_r.direct_lag, f_var.freq_az.direct_lag)).real
    # Calculate the phase shift
    angle_shift = find_phase_shift(var_uniform, var_filt)
    # Apply phase shift to re-center the data
    var_filt = var_filt.roll(az=int(var_filt.az.size * (angle_shift / 360)), roll_coords=True)

    # Fix the azimuthal coordinates after ifft to align with original azimuths
    var_filt = var_filt.assign_coords(az=az)
    # remove negative radial lags
    var_filt = var_filt.where(var_filt.r >= 0, drop=True)
    # Re-apply the original SNR mask (set the masked areas back to NaN)
    nr = var_filt.r.size
    var_filt = var_filt.where(snr_mask[:,:nr])

    return var_filt

# ------------------------------------------------
# colorbars functions

# snr
def snr_cmap():
    cdict11= {'red':  ((  0.0, 150/255, 150/255),
                    ( 2/19, 207/255, 207/255),
                    ( 6/19,  67/255,  67/255),
                    ( 7/19, 111/255, 111/255),
                    ( 8/19,  53/255,  17/255),
                    (11/19,   9/255,   9/255),
                    (12/19,     1.0,     1.0),
                    (14/19,     1.0,     1.0),
                    (16/19, 113/255,     1.0),
                    (17/19,     1.0,     1.0),
                    (18/19, 225/255, 178/255),
                    (  1.0,  99/255,  99/255)),

            'green': ((  0.0, 145/255, 145/255),
                    ( 2/19, 210/255, 210/255),
                    ( 6/19,  94/255,  94/255),
                    ( 7/19, 214/255, 214/255),
                    ( 8/19, 214/255, 214/255),
                    (11/19,  94/255,  94/255),
                    (12/19, 226/255, 226/255),
                    (14/19, 128/255,     0.0),
                    (16/19,     0.0,     1.0),
                    (17/19, 146/255, 117/255),
                    (18/19,     0.0,     0.0),
                    (  1.0,     0.0,     0.0)),

            'blue':  ((  0.0,  83/255,  83/255),
                    ( 2/19, 180/255, 180/255),
                    ( 4/19, 180/255, 180/255),
                    ( 6/19, 159/255, 159/255),
                    ( 7/19, 232/255, 232/255),
                    ( 8/19,  91/255,  24/255),
                    (12/19,     0.0,     0.0),
                    (16/19,     0.0,     1.0),
                    (17/19,     1.0,     1.0),
                    (18/19, 227/255,     1.0),
                    (  1.0, 214/255, 214/255))
            }
    cmap = colors.LinearSegmentedColormap(name='radar_NEXRAD_Zhh', segmentdata=cdict11)
    
    return cmap