from sm_scales_utils import *
import cartopy.geodesic as cgeo
from scipy.ndimage.measurements import label


### ! WARNING: THIS SCRIPT IS DESIGNED TO RUN ON MACHINES WHICH PERMIT SERIAL PARALLEL PROCESSING ! ###

###################################################################################
# - sim : control or sens, for 40 day control or 48hr restart sims respectively
# - expt : if sens, experiment to select. Options: control_48hr_runs ; wg_mcs ; large_only
# - day : if sens, day of experiment to sample (typically wish to split). Options 0 or 1.
# - mcs_time : time of storms for which cores will be identified. CURRENTLY FULL CODE ONLY WORKS FOR 17Z DESPITE INITIAL OPTIONS 
# - flt : Boolean - filtering removes orography, cores at back of storms, and storms initiated close to sample point. Recommended.
###################################################################################

# Filtering and other criteria chosen to align with Klein and Taylor, PNAS (2020) [KT2020]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=True) 
parser.add_argument("-e", "--expt", required=False, default="control_48hr_runs")
parser.add_argument("-d", "--day", required=False, type=int, default=0)
parser.add_argument("-mt", "--mcs_time", required=False, default=17)
parser.add_argument("-flt", "--filter", required=False, default=True)
args = parser.parse_args()

sim=args.sim
sim_day=args.day
expt=args.expt
if "control" in sim.lower():
    sim2="Control_run"
else:
    sim2=expt.capitalize()
    period=period[:-2]
    if expt!="control_48hr_runs":
        expt="sensitivity_runs_"+expt
    
times=args.mcs_time
if type(times)==int:
    hrs=[times]
    times="%02dZ"%times
elif times=="eve":
    hrsr=[17,18,19,20]
#This one does storm relative for any hour - track only (provided appropriate table available)
elif "storm" in times:
    samp_hr=int(times[:2])
    hrs=[samp_hr]
    
MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_evehours.csv".format(sim2))
full_tracks=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_rain.csv".format(sim2))
full_tracks["start_time"]=pd.to_datetime(full_tracks["start_time"])
#RESTRICT COMPOSITES TO SAHEL
MCS_data=MCS_data[(MCS_data.tminlat>9) & (MCS_data.tminlat<=19) & (MCS_data.tminlon>=348) & (MCS_data.tminlon<=378)]

if sim=="sens":
    MCS_data=MCS_data[MCS_data.sim_day==sim_day] 
    sim2=sim2+"_D%s"%(sim_day+1)

orog=xr.DataArray.from_iris(iris.load("/gws/nopw/j04/lmcs/hburns/Test_run/ancils/qrparm.orog")[9])


# Serial parallelisation - split period into 4 day chunks, specified by idx
def parallelise(idx):
    tab=[]
    period_segment=period[int(4*idx):int(4*(idx+1))]
    x=0

    for j, date in enumerate(period_segment):            
        date2=date+pd.Timedelta(sim_day,"d")
        print(date2)
        # Get relevant storms
        day_data=MCS_data[(MCS_data["day"]==date2.day) & (MCS_data["month"]==date2.month) & (MCS_data["hour"].isin(hrs))]

        # Get model 2D data fields needed for filtering and cores
        pre_bt=load_file(sim,date,15+sim_day*24,"surface_vars",stash_dict["toaolr"])[0,:,:]
        pre_bt=tb_from_olr(pre_bt)
        #pre_bt=pre_bt.where(pre_bt<241).fillna(0)

        mcs_bt=load_file(sim,date,17+sim_day*24,"surface_vars",stash_dict["toaolr"])[0,:,:]
        mcs_bt=tb_from_olr(mcs_bt)
        mcs_bt=mcs_bt.where(mcs_bt<241).fillna(0)

        try:
            w500=load_file(sim,date,17+sim_day*24,"model-diagnostics",stash_dict["omega_plevs"]).sel(PLEVS=500)[0,:,:]
        except:
            continue
        w500=w500.interp(longitude=mcs_bt.longitude,latitude=mcs_bt.latitude)

        # Now work through each storm
        for i in range(len(day_data)):
            storm=day_data.iloc[i]
            storm_core_locs=[]

            # Focus on big storms - consistent with KT2020
            if storm.area < 15000:
                continue
                
            bounds=[int(bound) for bound in storm["bounds"].split(",")]
            #Extra +1 in bounds selection needed to account for index selection offsets (otherwise breaks on 1-pixel wide storms) - requires lat/lon data
            fprint = mcs_bt[(bounds[1]-bounds[3]):bounds[1]+1,bounds[0]:(bounds[0]+bounds[2]+1)]
            # some storm objects from simpleTrack extend far beyond core Sahel domain (eg multiple apparently-merged storms)
            fprint = fprint.sel(latitude=slice(9,19),longitude=slice(348,378))

            # Get omega field in storm footprint
            wstorm=w500.where(fprint)
            thld=float(wstorm.quantile(0.005)) # low percentile as using omega and thus need the strongest negative values!
            # Find core objects where omega is sub-threshold:
            w_cores=wstorm.where(wstorm<thld).fillna(0)
            cores = label(w_cores.values)[0]

            # For each candidate core, find lat/lon location of centroid
            for id in np.unique(cores)[1:]:
                area=np.count_nonzero(cores==id)
                # Require mimimum 5-pixel cores
                if area<5:
                    continue

                idxs=np.nonzero(cores==id)
                # Get centroid indices; x is lon but coords lat/lon
                cx=(min(idxs[1]) + max(idxs[1]))/2
                cy=(min(idxs[0]) + max(idxs[0]))/2

                # Convert to lat/lon
                clon=w_cores.longitude[int(np.round(cx))]
                clat=w_cores.latitude[int(np.round(cy))]
                if float(clat) < 9 or float(clat) > 19 or float(clon) < 348 or float(clon) > 378:
                    print(date,storm.storm_id,float(clat),float(clon))
                storm_core_locs.append((float(clat),float(clon)))
            
            #Exception in case finds no objects
            if len(storm_core_locs)==0:
                continue

            
            # Apply filtering to core population to isolate LA interactions; consistent with KT2020
            if args.filter==True:
                filt_core_locs=[]
                for core_loc in storm_core_locs:
                    # Apply filtering to remove orographic influence: KleinTaylor20 filter out cores at altitude>450m
                    core_alt=orog.sel(latitude=slice(core_loc[0]-0.05,core_loc[0]+0.05),longitude=slice(core_loc[1]-0.05,core_loc[1]+0.05)).mean()
                    if core_alt>450:
                        continue

                    # Apply filtering to remove influence of cold cloud shields 2 hours prior
                    pre_mcs_bt=pre_bt.sel(latitude=slice(core_loc[0]-0.05,core_loc[0]+0.05),longitude=slice(core_loc[1]-0.05,core_loc[1]+0.05)).mean()
                    if pre_mcs_bt<241:
                        continue
                    
                    # Exclude MCSs initiated within 150km of core location (KT20200 use 300km - too strict as their tracks go down to much smaller cell sizes):
                    geo=cgeo.Geodesic()
                    if sim=="control":
                        storm_entry=full_tracks[full_tracks.storm_id==storm.storm_id]
                    else:
                        # Need to find relevant track from full_tracks dataset
                        id=np.where(full_tracks.values==storm.rain_max)
                        if len(id)>2:
                            id=[i for i in id[::2]]
                            id2=np.where(full_tracks.values==storm.rain_vol)
                            checks=[i for i in id2[::2]]
                            id=intersection(set(id), set(checks))                            
                        storm_entry=full_tracks.iloc[id[0]]

                    try:
                        start_lon, start_lat = ref_lons[int(storm_entry.clon_idx_00)], ref_lats[int(storm_entry.clat_idx_00)]
                        end_lon, end_lat = core_loc[1], core_loc[0]        
                        path=cgeo.Geodesic.inverse(geo,np.array((start_lon,start_lat)),np.array((end_lon,end_lat)))
                        if path[0][0]<=150e3: # units m
                            continue
                    except:
                        print(id,date)
                        pass

                    filt_core_locs.append(core_loc)
            
            else:
                filt_core_locs=storm_core_locs

            core_record=pd.DataFrame({"core_lon":[x[1] for x in filt_core_locs],"core_lat":[x[0] for x in filt_core_locs]})
            core_record["storm_id"]=storm.storm_id
            core_record["hour"]=storm.hour
            core_record["day"]=storm.day
            core_record["month"]=storm.month
            if sim=="sens":
                core_record["sim_day"]=storm.sim_day
            core_record["time"]=storm.time

            tab.append(core_record[core_record.columns.to_list()[::-1]])

    return pd.concat(tab,axis=0)

p = Pool(10)
part_tabs=p.map(parallelise,np.arange(10))
out=pd.concat(part_tabs,axis=0)

filtstr="NOfilt"
if args.filter:
    filtstr="filt"
out.to_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{}_Sahel_{}_cores.csv".format(sim2,times,filtstr),index=False)