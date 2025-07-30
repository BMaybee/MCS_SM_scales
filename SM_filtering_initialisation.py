import xarray as xr 
import wclass
import constants as cnst
import scipy.stats as stats
from scipy import ndimage
import numpy as np
import argparse
import datetime
import sys

###################################################################################
# - date: date at which to filter 06 UTC SM field, provide as 8 digit string %Y%m%d
# - scales : ["control","mcs_scales","large_only"] - for Control_48hr_runs, wg_mcs (SM(Large+Small)) and SM(LargeOnly) restarts respectively
# - method: ["wavelet+gaussian","gaussian"] - for SM(L+S) [i.e. wavelet + Gaussian smoothing] or SM(LO) [Gaussian only] respectively; control breaks beforehand, null operations
# - res : ["1p5km","4km"] - whether to filter high-res runs or 4km nest 
###################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, type=str)
parser.add_argument("-s", "--scales", required=True, type=str)
parser.add_argument("-m", "--method", required=False, default="wavelet+gaussian", type=str)
parser.add_argument("-r", "--res", required=False, default="1p5km", type=str)
args = parser.parse_args()

date_str=args.date
experiment=args.scales
methodg=args.method
res=args.res

# 06 UTC FIELD - filename timestamp 1 hour early.
soil_m=xr.open_dataset("/gws/nopw/j04/lmcs/u-cy045_control_run/u-cy045/{}T0000Z/Sahel/{}/RA3/um/soil_moistures_{}_T05.nc".format(date_str,res,date_str)).STASH_m01s08i223
dat = soil_m.isel(T1HR=0,ATM_SOIL=0)

if experiment=="control":
    print(date_str)
    soil_m.to_netcdf("/gws/nopw/j04/lmcs/bmaybee/Filtered_SM_initialisations/{}_05Z_SM_filter_control_ref.nc".format(date_str))
    sys.exit()

wObj = wclass.landwav('SM{}_control'.format(res))
wObj.read_img(dat.values, dat.grid_longitude_t.values, dat.grid_latitude_t.values)

if "wavelet" in methodg:
    coeffs, power, scales, period = wObj.applyWavelet(normed='none')
    variableIndi, scalesIndi = wObj.applyInverseWavelet(per_scale=True)
else:
    variableIndi = np.zeros((3,2260,3360)) # proxy data to avoid code breaking later. Data not used.
mask = np.isnan(dat.values)
dat_filled = dat.values.copy()
dat_filled[mask] = 0

if experiment=="mcs_scales":
    si = 0
    plus = 19 # corresponds to ~100km
    fact = 1
elif experiment=="large_only":
    si,plus = 0,0
    fact=0

# Extract relevant wavelet scale range and add to Gaussian smoothing of background state
# In ndimage.gaussian_filter, sigma=100 pixels -> 150km and truncate(=4)  x sigma = 400 pixel -> 600km smoothing radius
sm_filt = variableIndi[si:si+plus,:,:].sum(axis=0) * fact + ndimage.gaussian_filter(dat_filled, 100, mode='nearest') 
sm_filt[mask] = np.nan
sm_filt[sm_filt<0] = 0

output = np.expand_dims(sm_filt,(0,1))
# Concatenate filtered top layer SM field with unmodified lower-levels
output = xr.DataArray(output,coords=[soil_m.T1HR,soil_m.ATM_SOIL[:1],soil_m.grid_latitude_t,soil_m.grid_longitude_t])
output = xr.concat([output,soil_m[:,1:,:,:].drop_vars("ATM_SOIL_model_level_number")],dim="ATM_SOIL")
now=datetime.datetime.now()
output=output.assign_attrs(title="Filtered SM initialisation state",method=methodg,experiment="Exclude "+experiment.replace("_"," "),created_by="Ben Maybee, b.w.maybee@leeds.ac.uk",created_on=str(now))
if experiment=="mcs_scales":
    output.rename("STASH_m01s08i223").to_netcdf("/gws/nopw/j04/lmcs/bmaybee/Filtered_SM_initialisations/{}_05Z_SM_filter_wg_mcs.nc".format(date_str))
elif experiment=="large_only":
    output.rename("STASH_m01s08i223").to_netcdf("/gws/nopw/j04/lmcs/bmaybee/Filtered_SM_initialisations/{}_05Z_SM_filter_large_only.nc".format(date_str))

"""
#### FULL ROUND TRIP SANITY CHECK
wObj_afterv1 = wclass.landwav('SM{}_control'.format(res))
wObj_afterv1.read_img(sm_filt, dat.grid_latitude_t.values, dat.grid_longitude_t.values)
coeffsv1, powerv1, scalesv1, periodv1 = wObj_afterv1.applyWavelet(normed='scale')
new_perc1 = np.sum(powerv1[:,400:1200,:], axis=(1,2))/np.sum(powerv1[:,400:1200,:])

np.save('power_contribs_filt_test_V2.npy',new_perc1)
"""