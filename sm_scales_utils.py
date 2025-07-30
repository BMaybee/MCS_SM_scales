import xarray as xr
import pandas as pd
import numpy as np
import iris
import glob
import os
import datetime
import glob
import matplotlib.pyplot as plt
import argparse
import metpy.calc as mpcalc
import scipy.stats as stats
from metpy.units import units
from multiprocessing import Pool
import time

# Period covered by all simulations
period=pd.date_range(start="2006-07-25",end="2006-09-03")

# Data netcdfs label variables via UM stash code. Store here so don't need to use manually!
stash_dict={"sm":"STASH_m01s08i223", # soil_moistures
            "lhfx":"STASH_m01s03i234", # surface_vars
            "shfx":"STASH_m01s03i217", # surface_vars
            "t2":"STASH_m01s03i236", # surface_vars
            "td2":"STASH_m01s03i250", # surface_vars
            "q2":"STASH_m01s03i237", # surface_vars
            "lw_nsfc":"STASH_m01s02i201", # surface_vars
            "sw_nsfc":"STASH_m01s01i201", # surface_vars
            "tcc":"STASH_m01s09i217", # surface_vars
            "toaolr":"STASH_m01s02i205", # surface_vars
            "tcw":"STASH_m01s30i461", # surface_vars
            "mslp":"STASH_m01s16i222", # surface_vars
            "psfc":"STASH_m01s00i409", # surface_vars
            "precip":"STASH_m01s04i203", # surface_vars
            "pbl_depth":"STASH_m01s00i025", # surface_vars
            "q_plevs":"STASH_m01s30i205", # model-diagnostics
            "t_plevs":"STASH_m01s30i204", # model-diagnostics
            "u_plevs":"STASH_m01s30i201", # model-diagnostics
            "v_plevs":"STASH_m01s30i202", # model-diagnostics
            "omega_plevs":"STASH_m01s30i208", # model-diagnostics
            "z_plevs": "STASH_m01s16i202", # model-diagnostics
            "pv_plevs": "STASH_m01s15i229", # model-diagnostics
            "u_rho_levs": "STASH_m01s00i002", #3D_vars_hourly
            "v_rho_levs": "STASH_m01s00i003", #3D_vars_hourly
            "w_theta_levs": "STASH_m01s00i150", #3D_vars_hourly
            "pt_theta_levs": "STASH_m01s00i004", #3D_vars_hourly
            "p_theta_levs": "STASH_m01s00i408", #3D_vars_hourly
            "q_theta_levs": "STASH_m01s00i010", #3D_vars_hourly
            "traceRainEvap": "STASH_m01s00i714", #moisture-tracers
            "traceBL": "STASH_m01s00i706", #moisture-tracers
            "traceBLland": "STASH_m01s00i715", #moisture-tracers
            "shfx_con": "surface_upward_sensible_heat_flux",
            "td925": "dew_point_temperature",
            "convbt":"STASH_m01s02i205", # surface_vars
            "mcsbt":"STASH_m01s02i205" # surface_vars
           }

##############################
# Master function to load model data. Returns xr DataArray for single variable at specified hour within simulation, where 0 is 0UTC of day sim initialised.
#     - sim = ["control", "sens"]. Control here refers to continuous 40 day simulation. Sens is ANY group of separate 48hr sims, including split Controls
#     - date = pd.Timestamp or datetime.datetime format
#     - hour = any integer -24 <= h<= 54. 0 = 0 UTC on day specified, with all other values relative to this. Values >23 go into next day, etc. SENS SIMS THUS START AT 7! 
# To acquire the desired variable need two following parameters:
#     - file_list = ["soil_moistures","surface_vars","model-diagnostics","3D_vars_hourly","moisuture-tracers" [Control ONLY]]
#     - var = string STASH ID. Get this using stash_dict.
###############################

def load_file(sim,date,hour,file_list,var):
    date_str="2006%02d%02d" % (date.month,date.day)
    hour=hour-1 # all filenames are 1 hour BEHIND data timestamp
    if hour<=5 and sim=="sens": # have to return to control if sampling early pre-storm conditions in sens D1
        sim="control"
        if var=="STASH_m01s03i217": # shfx - this is missing from Control and thus extra work-arounds needed.
            file_list="sensible_hfx_control"
        
    date_str2=date_str
    if hour<0:
        hour=24+hour
        date=date-datetime.timedelta(days=1)
    else:
        if hour>23:
            hour=hour-24
            date=date+datetime.timedelta(days=1)
        if hour>23:
            hour=hour-24
            date=date+datetime.timedelta(days=1)
    exp_str="2006%02d%02d" % (date.month,date.day)

    if sim=="sens": # the simulation of interest (i.e. Control_48, SM(L+S), SM(LO) is set by global parameter expt.
        try:
            root_str=glob.glob("/gws/nopw/j04/lmcs/{}/{}T0600Z/*/{}T0600Z/Sahel/1p5km/RA3/um/".format(expt,date_str,date_str))[0]
        except:
            print(var,date_str,date_str2)
            #a=2/0
    elif sim=="control":
        root_str="/gws/nopw/j04/lmcs/u-cy045_control_run/u-cy045/{}T0000Z/Sahel/1p5km/RA3/um/".format(exp_str)

    if file_list=="sensible_hfx_control":
        ref_lats=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/sensible_heat_fluxes/20060727_sensible_heat_flux.nc").latitude
        #ref_lats=ref_lats.drop_vars("forecast_reference_time")
        data=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/sensible_heat_fluxes/{}_sensible_heat_flux.nc".format(exp_str))["surface_upward_sensible_heat_flux"
        ].isel(time=slice(hour%24,hour%24+1)) # time slice necessary to keep length 1 time axis.
        try:
            data=data.drop_vars("forecast_period")
            data=data.drop_vars("forecast_reference_time")
            data=data.assign_coords(latitude=ref_lats)
        except:
            pass
        return data[:,:2200,:3300]

    else:
        #print(root_str+"{}_{}_T{:02d}.nc".format(file_list,exp_str,hour))
        try:
            # THIS IS THE PRIMARY DATA LOAD USED IN >99% OF CALLS
            data=xr.open_dataset(root_str+"{}_{}_T{:02d}.nc".format(file_list,exp_str,hour))[var]

        # A few RARE instances of singular missing files can cause fatal errors - use previous timestep to sidestep these. Only really needed for Hovmoellers.
        except: 
            print("diverting",exp_str,hour,var)
            if hour>6: # going back to 5 will cause fatal errors again due to filename structure
                i=1
                while i>0:
                    try:
                        data=xr.open_dataset(root_str+"{}_{}_T{:02d}.nc".format(file_list,exp_str,hour-i))[var]
                        i=-10
                    except:
                        i+=1
            else:
                data=xr.open_dataset(root_str+"{}_{}_T{:02d}.nc".format(file_list,exp_str,hour+1))[var]

        # Remove annoying extra coordinates that can cause shape issues
        if file_list=="soil_moistures":
            data=data.isel(ATM_SOIL=0)
        try:
            data=data.isel(height_1_5m=0)
        except:
            pass
            
        # Frustrating quirk; files with instantaneous fields issued after restart contain an extra timestamp! Code always wants the latter as we are relying on namestamp being 1hr behind timestamps
        # Thus revert time axis so that index 0 is indeed the hour we asked for. Keep time dimension for ease of concatenation post load 
        if hour==0 or hour==6: 
            data=data.isel({data.dims[0]:slice(None,None,-1)})

        # For pressure level data, ditch 1000hPa values as usually sub-surface over Sahel; and swap order such that pressure coord descend (i.e. height ordered)
        try:
            ds=ds.sel(pressure=slice(None,950))
            ds=ds.isel(pressure=slice(None,None,-1))
        except:
            pass

        # Final formats: introduce common coordinate labels (xarray doesn't convert the different UM grid names); and restrict to common spatial domain to avoid overlap problems.
        return data[:,:2200,:3300].rename({data.dims[0]:"time",data.dims[-1]:"longitude",data.dims[-2]:"latitude"})



##############################################

# MCS tracking had to be done on slightly restricted domain due to simpleTrack requirements. Only numpy indices stored. Hence useful to have reference values.
ref_olr=xr.open_dataset('/gws/nopw/j04/lmcs/control_48hr_runs/20060725T0600Z/u-de766a/20060725T0600Z/Sahel/1p5km/RA3/um/surface_vars_20060726_T01.nc'
                       ).STASH_m01s02i205[:2200,:3300]
#ref_olr=xr.open_dataset("/work/scratch-pw3/hburns/surface_vars_20060727_T11.nc").STASH_m01s02i205[:2200,:3300]
ref_olr=ref_olr.assign_coords(longitude=ref_olr.grid_longitude_t-360)
ref_lons=ref_olr.grid_longitude_t
ref_lats=ref_olr.grid_latitude_t


# Conversion of TOAOLR to brightness temperature, via empirical correction of Yang and Slingo (2001)
def tb_from_olr(OLR):
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb

