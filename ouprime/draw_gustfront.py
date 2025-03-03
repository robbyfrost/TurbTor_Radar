# --------------------------------------------------
# Set up
# --------------------------------------------------
import pyart
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
from mpl_point_clicker import clicker
from matplotlib.ticker import MultipleLocator
sys.path.append('/Users/robbyfrost/Documents/MS_Project/TurbTor_Radar/')
from functions import *
import os
from scipy.interpolate import interp1d

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')
# --------------------------------------------------
# Read radar data
# --------------------------------------------------
dir_rad = '/Users/robbyfrost/Documents/MS_Project/data/OU-PRIME/20100510/proc/'

rall = []
for file in sorted(os.listdir(dir_rad)):
    if file.startswith('nexrad'):
        file_path = f"{dir_rad}/{file}"
        radar = pyart.io.read(file_path)
        rall.append(radar)

# Calculate vorticity on each sweep
vall = []
for i, radar in enumerate(rall):
    vortz_full = np.zeros_like(radar.fields['velocity']['data'])
    for swp in range(radar.nsweeps):
        # extract sweep
        radswp = radar.extract_sweeps([swp])
        # extract fields
        az = radswp.azimuth['data']
        el = radswp.elevation['data']
        r = radswp.range['data']
        vr = radswp.fields['velocity']['data']
        # smooth vr
        vr_da = xr.DataArray(vr, dims=["az", "r"])
        vr = vr_da.rolling(az=3,r=3, center=True).mean().values
        # calculate vorticity
        vortz = np.zeros_like(vr)
        vortz[1:, :] = ( (vr[1:,:] - vr[:-1,:]) / (np.deg2rad(az[1:]) - np.deg2rad(az[:-1]))[:,np.newaxis] ) * (1 / r)
        # add to sweep
        vortz_field = {
            'data': vortz,
            'units': '/s',
            'long_name': 'Inferred Vertical Vorticity',
            'standard_name': 'Inferred vertical vorticity',
        }
        radswp.add_field('vortz', vortz_field)
        # combine sweeps
        if swp == 0:
            azidx = az.size
            vortz_full[:azidx,:] = vortz
        else:
            vortz_full[azidx:azidx+az.size,:] = vortz
            azidx = azidx + az.size
    # add to radar object
    vortz_full_field = {
        'data': vortz_full,
        'units': '/s',
        'long_name': 'Inferred Vertical Vorticity',
        'standard_name': 'Inferred vertical vorticity',
    }
    radar.add_field('vortz', vortz_full_field)

# --------------------------------------------------
# Function to plot radial velocity
# --------------------------------------------------

def plot_vr(swp, tidx):
    radar = rall[tidx]

    display = pyart.graph.RadarDisplay(radar.extract_sweeps([swp]))

    fig, ax = plt.subplots(figsize=(10,10), ncols=1)

    # radial velocity
    vmin, vmax = -50, 50
    display.plot_ppi('velocity', ax=ax, 
                    vmin=vmin, vmax=vmax,
                    cmap="Carbone42",
                    colorbar_flag=False,
                    title_use_sweep_time=True,
                    axislabels_flag=False)
    cbar1 = fig.colorbar(display.plots[0], ax=ax, aspect=20, pad=0.03, fraction=0.06, shrink=0.79)
    cbar1.set_label("$V_r$ [m s$^{-1}$]")
    cbar1.set_ticks(np.arange(vmin, vmax + 0.001, 10))

    # clean up plot
    ax.set_aspect('equal')
    ax.set_xlim(-15,25)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("Zonal Distance [km]")
    ax.set_ylim(0,40)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Meridional Distance [km]")
    ax.grid(alpha=0.6)
    fig.tight_layout()

    return fig, ax

# --------------------------------------------------
#  Draw gust fronts
# --------------------------------------------------
fig, ax = plot_vr(0, 1) 

klicker = clicker(
    ax, 
    ["RFGF"], 
    markers=["*"])
plt.show()
coords = klicker.get_positions()

# --------------------------------------------------
# Create gust front line and make relative ranges
# --------------------------------------------------
# Extract coordinates
xRFGF, yRFGF = coords["RFGF"][:,0], coords["RFGF"][:,1]
# Sort points by xRFGF
sorted_indices = np.argsort(xRFGF)
xRFGF_sorted = xRFGF[sorted_indices]
yRFGF_sorted = yRFGF[sorted_indices]
# Create interpolation function
interp_fn = interp1d(xRFGF_sorted, yRFGF_sorted, kind='linear')
# Create a dense array of x values for the line
x_dense = np.linspace(xRFGF_sorted[0], xRFGF_sorted[-1], num=100)
y_dense = interp_fn(x_dense)

# radar fields, cartesian grid
r = radar.range['data']
az = radar.get_azimuth(0)
nr, naz = r.size, az.size
print(naz)
XX, YY = dis_angle_to_2Dxy(r, north0_to_arctheta(az))
XX, YY = XX.T/1e3, YY.T/1e3

# match line to radar gates
XXf, YYf = XX.ravel(), YY.ravel()  # Flatten XX into a 1D array
xidx = np.abs(XXf[:,None] - x_dense[None,:]).argmin(axis=0)  # Find closest indices
xidx = np.unravel_index(xidx, XX.shape)
yidx = np.abs(YYf[:,None] - y_dense[None,:]).argmin(axis=0)  # Find closest indices
yidx = np.unravel_index(yidx, YY.shape)

# Array of ranges relative to RFGF
ranges = np.tile(r, (naz, 1))