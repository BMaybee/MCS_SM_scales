import xarray as xr
import pandas as pd
import glob
import datetime
import argparse
import numpy as np
import cartopy.crs as ccrs
import scipy.stats as stats
import cartopy.geodesic as cgeo
from itertools import groupby
import ast
import warnings
from scipy.ndimage import label
import time
from multiprocessing import Pool
warnings.filterwarnings("ignore")

# Function to compute empirical TOA BT (Yang and Slingo, 2001) from OLR
def olr_to_bt(olr):
    #Application of Stefan-Boltzmann law
    sigma = 5.670373e-8
    tf = (olr/sigma)**0.25
    a = 1.228
    b = -1.106e-3
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb

##############################
# simpleTrack produces hourly text files for detected snapshots. Use global dictionary to collate information from the .txt files on each tracked storm_id
# Note this function does NOT gather information about parent/child storms, which is gathered by simpleTrack. Information garnered:
#    - Snapshot area (in pixels), centroid and bounding box (specified as indices on tracked numpy arrays)
#    - Instantaneous object velocity (dx, dy)
#    - Extreme and mean of the tracked quantity (so brightness temp) over snapshot footprint. 
def get_track_info(dict,sim,tstamp):
    #Reads from IMAGES_DIR in simpleTrack wrapper.py. File format set by tracker algorithm.
    ffile="/gws/nopw/j04/lmcs/bmaybee/storm_track_data/{}/history_S100_T241_A444_{}.txt".format(sim,tstamp)
    if tstamp[-4:-2]=="00":
        print(ffile)
    with open(ffile,"r") as data_file:
        for line in data_file:
            data = line.split()
            # There are some lines which are not storm entries ; filter out
            if len(data) > 5:
                storm_id=data[1]
                if storm_id in dict.keys():
                    pass
                else:
                    dict[storm_id]=[storm_id,tstamp]
                dict[storm_id].append(int([d for d in data if d.startswith('area=')][0].replace('area=','')))
                dict[storm_id].append([d for d in data if d.startswith('box=')][0].replace('box=',''))
                dict[storm_id].append(float([d for d in data if d.startswith('centroid=')][0].replace('centroid=','').split(',')[0]))
                dict[storm_id].append(float([d for d in data if d.startswith('centroid=')][0].replace('centroid=','').split(',')[1]))
                #Multiplication factor converts rawstorm speeds from pixels/timestep -> km/h:
                dict[storm_id].append(1.5*float([d for d in data if d.startswith('dx=')][0].replace('dx=','')))
                dict[storm_id].append(1.5*float([d for d in data if d.startswith('dy=')][0].replace('dy=','')))
                dict[storm_id].append(float([d for d in data if d.startswith('extreme=')][0].replace('extreme=','')))
                dict[storm_id].append(float([d for d in data if d.startswith('meanv=')][0].replace('meanv=','')))
    return dict


# Wrapper for get_track_info - collate storm track info into DataFrame in which each row is one storm's full track.
# Beginning of each row is the start time. Each hour of storm's life then covered by grouping of eight columns, specified by number of hours since initiation.
# Advantages: facilitates lifetime statistics and later calculations; minimises NaN padding
# Disadvantage: can be difficult to group storms for a given timestamp
def build_table(sim,mcs):
    print(sim)
    
    dict={}
    for i, tstamp in enumerate(period):
        stamp_str="%04d%02d%02d_%02d00"%(tstamp.year,tstamp.month,tstamp.day,tstamp.hour)
        try:
            dict=get_track_info(dict,sim,stamp_str)
        except:
            pass
        
    #Dictionary built, saving to DataFrame:
    storms=pd.DataFrame.from_dict(dict,orient="index")
    print(storms)
    hrs=(storms.shape[1] - 2)/8
    col_names=["area","bounds","clon_idx","clat_idx","PSu","PSv","tmin","tmean"]
    col_names=["storm_id","start_time"]+[col_name+"_%02d"%i for i in range(0,int(hrs)) for col_name in col_names]
    storms.columns=col_names
    storms=storms.set_index(storms["storm_id"])
    del storms["storm_id"]
    #Apply MCS criteria:
    if mcs=="standard":
        storms["tmin_min"]=storms.filter(regex="tmin").min(axis=1)
        
        pixel_thld=5000/(1.5**2)
        areas=storms.filter(regex="area").astype("float")
        storms["mcs_thld"]=areas.where(areas>=pixel_thld).count(axis=1)
        
        #mcs_data=storms[(storms["tmin_min"]<223) & (storms["mcs_thld"]>=1)] # - rather than apply thresholds now, do later so can compare with full convective storm population
        storms.to_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_track_output.csv".format(sim),index=False)
    if mcs=="MCSMIP":
        storms["tmin_min"]=storms.filter(regex="tmin").min(axis=1)
        
        pixel_thld=40000/(1.5**2)
        areas=storms.filter(regex="area").astype("float")
        storms["mcs_thld"]=areas.where(areas>=pixel_thld).count(axis=1)
        
        mcs_data=storms[(storms["tmin_min"]<227) & (storms["mcs_thld"]>=4)]
        mcs_data.to_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_track_output.csv".format(sim),index=False)
    if mcs==None:
        mcs_data.to_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_all_track_output.csv".format(sim),index=False)
##############################

##############################
# simpleTrack (as deployed) only tracks a single field - brightness temperature used to get cloud features, but also want MCS rainfall information.
# This function cycles through the detected snapshots; isolates rainfall field under footprint; and extracts storm total and max rainfall on 0.1 degree grid
# Can be passed a df directly (mcs_tab); and used to output netcdf of tracked MCS footprints (mask_out)
def pop_rains_masks(sim,mcs_tab=[],mask_out=False):
    if len(mcs_tab)==0:
        mcs_tab=pd.read_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_track_output.csv".format(sim))
        mcs_tab["start_time"]=pd.to_datetime(mcs_tab["start_time"].astype(str),format="%Y%m%d_%H%M")
    mcs_tab=mcs_tab[(mcs_tab["tmin_min"]<223) & (mcs_tab["mcs_thld"]>=1)]
    #mcs_tab["storm_id"]=np.arange(1,len(mcs_tab)+1)
    mcs_tab.insert(0,"storm_id",np.arange(1,len(mcs_tab)+1))
    period=pd.date_range("2006-07-25 01:00","2006-09-04 00:00",freq="H")

    # Key lists to populate.
    # - keeps track of the mask dataArrays. Split on the timesteps within period. After initial creation a mask will be accessed and updated multiple times, from storms with later init times.
    mcs_masks=[]
    # - populates rain columns in the collated csv files. Split on start_time
    rain_vals=[]

    # Loop through unique start times
    for start_time in pd.to_datetime(mcs_tab.start_time.unique()):
        #Isolate storms which all initiated at same time:
        mcs_start_group=mcs_tab[mcs_tab["start_time"]==start_time].dropna(axis=1,how="all")
        print(start_time,": ",len(mcs_start_group)," storms") 
        
        # hrs eqn gets number of distinct hour timesteps in grouping. The OLR output csvs columns comprise:
        # - groups of 8 columns per timestep onward from different possible initialisation times
        # - 4 extra columns: id, start_time, min storm T and mcs_thld (# times storm breached OLR size threshold)
        hrs=(mcs_start_group.shape[1] - 4)/8
        # Now loop through the lifetimes of the storms
        for hr in range(int(hrs)):
            #if hr % 3 == 0:
            #   print(hr)
            rain_vols,rain_maxes=np.zeros(len(mcs_start_group)),np.zeros(len(mcs_start_group))
            loc=mcs_start_group.columns.to_list().index("tmin_%02d"%hr)

            #Isolate single timestep, hr hours onward from common start time:
            mcs_timestep=mcs_start_group.filter(regex="_%02d"%hr)
            timestamp=start_time+datetime.timedelta(hours=(hr))
            #Get location in period:
            period_idx=int(np.where(period==timestamp)[0])
            tstr=timestamp-datetime.timedelta(hours=1) # filenames are one hour behind actual data time.
            tstr="%04d%02d%02d_T%02d.nc"%(tstr.year,tstr.month,tstr.day,tstr.hour)
            if sim=="Control_run":
                sfile=root+"{}T0000Z/Sahel/1p5km/RA3/um/surface_vars_{}".format(tstr[:8],tstr)
            else:
                """ # code for midway-reinit style experiments
                simhr=(timestamp - date).total_seconds()/3600
                if simhr<24:
                    exp_str1=exp_str
                else:
                    exp_day=mcs_tab["start_time"].iloc[0]+datetime.timedelta(days=1)
                    exp_str1="%04d%02d%02dT0600Z"%(exp_day.year,exp_day.month,exp_day.day)
                    """
                sfile=root+"{}/Sahel/1p5km/RA3/um/surface_vars_{}".format(exp_str,tstr) # exp_str1 changed -> exp_str

            try:
                # Protection from broken files (RARE); known issue with 20060827_T01 in control
                precip=xr.open_dataset(sfile).STASH_m01s04i203[0,:,:]
                rfile=xr.open_dataset(sfile).STASH_m01s02i205[0,:,:]
            except:
                print("Fail, ", sfile)
                continue
            bt=olr_to_bt(rfile)
            bt_mask=bt.where(bt<241).fillna(0)

            # Add new set of labelled data - always done if not generating mask (memroy intensive)
            if period_idx >= len(mcs_masks) or mask_out==False:
                # Label all distinct features meeting criteria. Structure is crucial to match tracker output accurately.
                labeled_array = label(bt_mask.values,structure=np.ones((3,3)))[0]
                print(period[period_idx],len(mcs_masks))
                
            # When generating mask, if timestep has been encountered previously, load in the relevant masks, which are then edited.
            else:
                #print(period_idx,period[period_idx])
                bt_mask=mcs_masks[period_idx]
                labeled_array=bt_mask.values

            for i in range(len(mcs_start_group)):
                # Get storm bounds - output direct from simpleTrack. Exception handles empty cell values, i.e. where storm has dissipated
                try:
                    bounds=[int(bound) for bound in mcs_timestep["bounds_%02d"%hr].iloc[i].split(",")]
                except:
                    rain_vols[i],rain_maxes[i]=np.NaN,np.NaN
                    continue
                    
                id=mcs_timestep.index[i]
                # Get the region of mask which corresponds to storm "id". (0,1) gives top LEFT coordinate of domain; 2 gives width; 3 gives height (l572 of object_tracking.py).
                # Thus require lat/lon form data
                storm_zoom = labeled_array[(bounds[1]-bounds[3]):bounds[1]+1,bounds[0]:(bounds[0]+bounds[2])+1]
                # Get most common non-zero label in storm_zoom:
                lab=stats.mode(np.where(storm_zoom!=0,storm_zoom,np.NaN),axis=None,nan_policy="omit",keepdims=True)[0]
                
                # There are some instances where a bound width/height = 0, so no result in lab. Catch prevents a fatal error.
                # Investigations of case this found on (UM 23/08/16 03:00) showed was embeded in another feature, so don't record a storm area.
                if len(lab)>0:
                #if storm_zoom.size > len(storm_zoom):
                    # Catch for just in case lab does not identify a storm label; mode on all nans gives result 0
                    if lab==0:
                        rain_vols[i],rain_maxes[i]=np.NaN,np.NaN
                        print("No storms")
                    else:
                        # Replace identified label with storm id - LARGE PAD; crucial to avoid mix up with the labels set in l143. Being -ve enables easy extraction of storm_ids later on
                        # In this manner labeled_array builds up pictures of our storms if needed
                        labeled_array=np.where(labeled_array==lab,-9999999+id,labeled_array)
                        # Get the rainfall values within the identified storm area:
                        if len(labeled_array[labeled_array==-9999999+id].flatten()) == 0:
                            print("No labels found, error",hr,lab,id)
                            print(period_idx)
                            break
                        storm_precip=3600*precip.where(np.where(labeled_array==-9999999+id,1,0)>0)
                        # Extract total and max storm rainfall at this timestep; conversion to mm/hr included above. Interpolation conducted to avoid crazy maxima.
                        rain_vols[i]=float(storm_precip.sum())
                        lat,lon=ref_lats[(bounds[1]-bounds[3]):bounds[1]+1], ref_lons[bounds[0]:(bounds[0]+bounds[2])+1]
                        storm_precip=storm_precip.interp(grid_latitude_t=np.arange(min(lat),max(lat)+0.1,0.1),grid_longitude_t=np.arange(min(lon),max(lon)+0.1,0.1))
                        rain_maxes[i]=float(storm_precip.max())
                        #print(float(storm_precip.sum()),float(storm_precip.max()))
                else:
                    print("Missing dimension")
                    rain_vols[i],rain_maxes[i]=np.NaN,np.NaN
                

            # Add rain values into the csv component:
            mcs_start_group.insert(loc+1,"rain_vol_%02d"%hr,rain_vols)
            mcs_start_group.insert(loc+2,"rain_max_%02d"%hr,rain_maxes)

            if mask_out:
                # Get storm masks into a dataArray, then output appropriately based on whether new or previous time.
                bt_mask.values=labeled_array
                if period_idx >= len(mcs_masks):
                    mcs_masks.append(bt_mask.rename("mcs_mask"))
                else:
                    mcs_masks[period_idx]=bt_mask
        #print(mcs_start_group)
        #print(mcs_start_group.filter(regex="rain"))
        rain_vals.append(mcs_start_group)

        # save information for safety when doing full 40 day initial control run, in case of slurm breakage
        if start_time.hour==23 and sim=="control_run":
            out=pd.concat(rain_vals,axis=0)
            out.to_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/MCS_track_output_rains_CRUN{}.csv".format(start_time.dayofyear),index=False)
    #print(mcs_start_group)
    
    if mask_out:
        mcs_masks=xr.concat(mcs_masks,dim="time")
        # Removes all remnant non-MCS label items and then corrects for the large padding:
        mcs_masks=mcs_masks.where(mcs_masks<0) + 9999999
        mcs_masks=mcs_masks.assign_attrs(units="unitless",long_name="MCS mask with track number")
        mcs_masks.to_netcdf("/gws/nopw/j04/lmcs/bmaybee/storm_track_data/Tables/{}_{}_MCS_track_masks.nc".format(sim))
    
    rain_vals=pd.concat(rain_vals,axis=0)
    rain_vals[["temp1","temp2"]]=rain_vals[["tmin_min","mcs_thld"]]
    rain_vals=rain_vals.drop(columns=["tmin_min","mcs_thld"]).rename(columns={"temp1":"tmin_min","temp2":"mcs_thld"})
    rain_vals.to_csv("LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_rain.csv".format(sim),index=False)
    
    return rain_vals
##############################


# simpleTrack required slightly restricted grid - useful to have reference to the tracked grid (cut off far N and E).
ref_olr=xr.open_dataset('/gws/nopw/j04/lmcs/u-cy045_control_run/u-cy045/20060725T0000Z/Sahel/1p5km/RA3/um/surface_vars_20060725_T01.nc'
                       ).STASH_m01s02i205
ref_olr=standardise(ref_olr)[:2200,:3300]
ref_lons=ref_olr.longitude
ref_lats=ref_olr.latitude


#####################################################################
# - sim specifies scale experiment : control_run (continous 40 days) ; control_48hr_runs (2 day Controls used in paper); sensitivity_runs_wg_mcs ; sensitivity_runs_large_only
# - index : use to select individual experiments from 2-day restarted runs (i.e. any except control_run). Value corresponding to index of expt within 40 day period.
# - mcs : selects criteria for keeping storm track in initial collation. Options standard, MCSMIP
# - update : default script action is to collate simpleTrack .txt outputs into csv. Updates further populate the csv:
#      - rains: finds rainfall values (volume, max) for every storm track entry. Also outputs details of storms at peak convection hours (16-22UTC)
#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=False, type=str, default="control_run")
parser.add_argument("-i", "--index", required=False, type=int)
parser.add_argument("-mcs", "--mcs", required=False, default="standard")
parser.add_argument("-u", "--update", required=False, type=str)
args = parser.parse_args()

start=time.time()
sim=args.sim

if sim.lower()=="control" or sim.lower()=="control_run":
    sim="control_run"
    root='/gws/nopw/j04/lmcs/u-cy045_control_run/u-cy045/'
    period=pd.date_range(start="2006-07-25 01:00",end="2006-09-04 00:00",freq="H")
    date=period[0]
else:
    runs=pd.date_range(start="2006-07-25 07:00",end="2006-09-01 07:00",freq="D")
    date=runs[args.index]
    exp_str="%04d%02d%02dT0600Z" % (date.year,date.month,date.day)
    sim=sim+"/"+exp_str

    if sim=="control_48hr_CHECK/20060725T0600Z":
        root='/gws/nopw/j04/lmcs/u-cy045_control_run/u-cy045/'
        sim="control_48hr_CHECK"
    else:
        root=glob.glob('/gws/nopw/j04/lmcs/{}/*'.format(sim))[0]+'/')
    print(root)
    period=pd.date_range(start=date,end=date+datetime.timedelta(hours=48),freq="H")

if args.update is None:
    build_table(sim,args.mcs)
    
elif "rains" in args.update:
    pop_rains_masks(sim)
