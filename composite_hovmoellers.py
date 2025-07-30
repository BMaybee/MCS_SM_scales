from sm_scales_utils import *

### ! WARNING: THIS SCRIPT IS DESIGNED TO RUN ON MACHINES WHICH PERMIT SERIAL PARALLEL PROCESSING ! ###

#### Main function for extracting spatial composite of field about point:
# - field : dataArray for variable, with coordinates in form ([vertical_coord],latitude,longitude), where lat and lon are both ascending, and vertical_coordinate is optional
# - clat, clon : the sample point coordinates
# - size : pixel size of latitudinal section. For 1.5km grid -> 1.5x<size> section. For longitude, we take 150km centred-section, i.e +/- 75 either side (50 pixels)
# Returns numpy array.
def composite_section(field,clat,clon,size=801):
    s=len(field.shape)
    if s==2:
        field=field.expand_dims("pressure",axis=0)
    #ll_border=size/100
    ll_border=6 #3
    field_comp=field.sel(latitude=slice(clat-ll_border,clat+ll_border),
                         longitude=slice(clon-1,clon+1))
    l1,l2=field_comp.shape[1],field_comp.shape[2]
    field_comp=field_comp[:,int(l1/2 - size/2):int(l1/2 + size/2 + 2),int(l2/2 - 50):int(l2/2 + 50 + 2)] # facts of 2 are for padding rounding errors; need to ensure all lat composites exactly the same size.
    field_comp=field_comp[:,:size,:101].mean(dim="longitude")
    if field_comp.shape[1] != size:
        print(field.name,field_comp.shape,clat,clon)
    if s==2:
        field_comp=field_comp.isel(pressure=0)
    return field_comp.values

psize=10 # Pool size

###################################################################################
# - sim : ["control","sens"] - for continuous 40 day Control, or ANY set of 48hr simulations, respectively. All paper results use Sens.
# - expt : if sens, sensitivity experiment to select. Options: ["control_48hr_runs", "wg_mcs", "large_only"]
# - day : if sens, day of experiment to sample (typically wish to split). Options 0 or 1, i.e. pythonic indexing.
# - method : specification of the type of sampling point being composited. 
#         - Options ["core","corePRE","diffPmax","diffPmaxWET"] (last two being "MesoDRY" and "MesoWET" SM patches, first MCS cores).
#                 - "corePRE" is option for sampling 40-day-Control fields pre 07Z for Sens MCSs
#         - "core<>" and "diffPmax<>" lead to different temporal Hovmoellers; core finds fields BEFORE loc_time; diffPmax finds fields AFTER loc_time.
# - loc_time : the hour at which sampling points have been identified, UTC. Default 17 for "core", 9 for SM patches
# - fields : the group of fields for which to calculate composites
#     - "2dfields" : lat-time Hovmoeller of anomalous and absolute SM, q925, q800, T925, u925, v925, u650, u650-u925 shear, 925hPa horizontal divergence [no anom], 
#                    TCW, TCC, rainfall [no anom], PBL depth, 925hPa dewpoint [no anom], 925hPa equivalent potential temperature, CAPE proxy [no anom], 925hPa potential temperature.
#     - "hfxs" : lat-time Hovmoeller of anomalous and absolute surface sensible hfx, latent lhfx, available energy, evap frac, net SW, net LW.
#     - "moisture_fluxes" : lat-time Hovmoeller of absolute 925hPa and column-integrated moisture flux divergence, zonal moisture flux, meridional moisture flux
#     - "instability" : time-series of ICAPE, 925hPa CAPE, 925hPa CIN and LCL calculated from 1 degree mean profiles
# - split : integer. Useful as these computations quickly get memory intense, even on big machines (2TB). This parameter enables further 
#           parallelisation by splitting the workflow into 4 complementary runs. Designed to be used with slurm sbatch_array
###################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=True) 
parser.add_argument("-e", "--expt", required=False, default="control_48hr_runs")
parser.add_argument("-d", "--day", required=False, type=int, default=0)
parser.add_argument("-m", "--method", required=False, default="core")
parser.add_argument("-lt", "--loc_time", required=False, default=17, type=int)
parser.add_argument("-f", "--fields", required=False, default="soil_moisture")
parser.add_argument("-flt", "--filter", required=False, default=True)
parser.add_argument("-split","--split",required=False,type=int)
args = parser.parse_args()

sim=args.sim.lower()
expt=args.expt
sim_day=args.day
if "control" in sim:
    sim2="Control_run"
    period=period[:-2]
else:
    period=period[:-2]
    sim2=expt.capitalize()+"_D%s"%(sim_day+1)
    if expt!="control_48hr_runs":
        expt="sensitivity_runs_"+expt

if args.split is not None:
    period=period[args.split*12:(args.split+1)*12]
    if len(period)<12:
        psize=1
    else:
        psize=3
        
method=args.method
loc_hr=args.loc_time
fields=args.fields

#### Get sample points ####
# SM patches
if "diffPmax" in method:
    MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/Wg_mcs_D1_sm-diff_{:02d}Z_pmax_locs.csv".format(loc_hr))
    if method=="diffPmax":
        MCS_data=MCS_data[MCS_data["diff_sign"]==1] # want to sample around DRY patches in control
    elif method=="diffPmaxWET":
        MCS_data=MCS_data[MCS_data["diff_sign"]==-1]
    loc_x,loc_y="pmax_lon","pmax_lat"

# MCS cores
else:
    if loc_hr==17:
        MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{:02d}Z_Sahel_filt_cores.csv".format(sim2,loc_hr))
        loc_x,loc_y="core_lon","core_lat"
    else:
        MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_comphours.csv".format(sim2))
        full_tracks=glob.glob("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_rain.csv".format(sim2))
        full_tracks["start_time"]=pd.to_datetime(full_tracks["start_time"])

    if sim=="sens":
        MCS_data=MCS_data[MCS_data.sim_day==sim_day]
        
#RESTRICT COMPOSITES TO SAHEL
MCS_data=MCS_data[(MCS_data[loc_y]>9) & (MCS_data[loc_y]<=19) & (MCS_data[loc_x]>=348) & (MCS_data[loc_x]<=378)]

if fields=="2dfields":
    field_vars=["sm","q925","q800","t925","u925","v925","ushear650_925","horizontal_divergence",
                "tcw","tcc","precip","pblh","td925","theta_e925","capeprox","pt925"]
elif fields=="hfxs":
    field_vars=["sw_nsfc","lw_nsfc","shfx","lhfx","ae","ef"]
elif fields=="moisture_fluxes":
    field_vars=["MFD_925","MF_u925","MF_v925","MF_u","MF_v","MFD"]#
    lid=11
elif fields=="instability":
    field_vars=["icape","cape","cin","lcl"]

try:
    print(field_vars)
except:
    raise("Error: invalid field set entered")



#########################################################
# KEY BLOCK - sets up timeframe over which Hovmoeller constructed.
# 0 = 00UTC day of storm/diffPmax sampling point.

if method=="core":
    if sim_day==0:
        samp_period=np.arange(7,loc_hr+1)
    elif sim_day==1:
        samp_period=np.arange(-17,loc_hr+1)
elif method=="corePRE":
    samp_period=np.arange(-18,7)
elif "diffPmax" in method:
    samp_period=np.arange(loc_hr,54)
#########################################################


# Function to get time-series of instability measures. Distinct to parallelise, does not use composite_section.
# Still used as a serial-parallel function. Done on dates within period. Split period into 10 groups of 4 days. Idx specifies the group.
def instability_evolution(idx):
    comp={}
    for field in field_vars:
        comp[field]=np.zeros(len(samp_period))
        comp[field+"_count"]=np.zeros(len(samp_period))
    # Segmentation of full period for serial paraellelisation
    period_segment=period[int(4*idx):int(4*(idx+1))]
    
    for j, date in enumerate(period_segment):
        print(date)
        date2=date+pd.Timedelta(sim_day,"d")
            
        day_data=MCS_data[(MCS_data["day"]==date2.day) & (MCS_data["month"]==date2.month) & (MCS_data["hour"]==loc_hr)]
        data_dict={}

        for hr_idx, samp_hr in enumerate(samp_period):
            print(samp_hr)
            if samp_hr<=0 and date==pd.Timestamp("2006-07-25"):
                continue

            t=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["t_plevs"])[0,:,:,:]
            q=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["q_plevs"])[0,:,:,:]
            t=t.where(t!=0)
            q=q.where(q!=0)

            for core_loc in day_data[[loc_y,loc_x]].values:
                # Get 1 degree mean profiles about sample point
                prof_t=t.sel(latitude=slice(core_loc[0]-0.5,core_loc[0]+0.5),longitude=slice(core_loc[1]-0.5,core_loc[1]+0.5)).mean(dim=["latitude","longitude"])
                prof_t=prof_t.dropna(dim="PLEVS")
        
                prof_q=q.sel(latitude=slice(core_loc[0]-0.5,core_loc[0]+0.5),longitude=slice(core_loc[1]-0.5,core_loc[1]+0.5)).mean(dim=["latitude","longitude"])
                prof_q=prof_q.dropna(dim="PLEVS")
                prof_p=prof_q.PLEVS

                # For profile get dewpoint and LCL.
                prof_td=mpcalc.dewpoint_from_specific_humidity(prof_p*units("hPa"),prof_t*units("K"),prof_q*units("kg/kg"))
                lcl=mpcalc.lcl(prof_p[0]*units("hPa"),prof_t[0]*units("K"),prof_td[0])
                #print(lcl)


                # To calculate ICAPE we need to store values of CAPE calculated at each pressure level. ALso store ICAPE
                cape_cin=np.zeros((len(prof_p),2))
                
                data_dict={}
                # Check below LCL
                if prof_p[0] > lcl[0].magnitude and prof_t[0] > lcl[1].magnitude:
                    parc=mpcalc.parcel_profile(prof_p*units("hPa"),prof_t[0]*units("K"),prof_td[0])
                    cpe=mpcalc.cape_cin(prof_p*units("hPa"),prof_t*units("K"),prof_td,parc)
                    cape_cin[0,0]=cpe[0].magnitude
                    cape_cin[0,1]=cpe[1].magnitude
                    data_dict["cape"]=cpe[0].magnitude
                    data_dict["cin"]=cpe[1].magnitude
                    data_dict["lcl"]=lcl[0].magnitude
                                    
                    for i, p in enumerate(prof_p.values[1:]):
                        # Calculate CAPE when below LCL - doesn't exist otherwise.
                        if prof_p.sel(PLEVS=p) > lcl[0].magnitude and prof_t.sel(PLEVS=p) > lcl[1].magnitude:
                            parc=mpcalc.parcel_profile(prof_p.sel(PLEVS=slice(p,None))*units("hPa"),prof_t.sel(PLEVS=p)*units("K"),prof_td.sel(PLEVS=p))
                            cpe=mpcalc.cape_cin(prof_p.sel(PLEVS=slice(p,None))*units("hPa"),prof_t.sel(PLEVS=slice(p,None)).sel(
                                                PLEVS=slice(p,None))*units("K"),prof_td.sel(PLEVS=slice(p,None)),parc)
                            cape_cin[i,0]=cpe[0].magnitude
                            cape_cin[i,1]=cpe[1].magnitude
                        else:
                            continue
                    data_dict["icape"]=-1/9.81 * np.trapz(cape_cin[:,0],x=100*prof_p.values)
                else:
                    data_dict["cape"],data_dict["cin"],data_dict["lcl"],data_dict["icape"]=0,0,0,0
                
                for var in field_vars:
                    comp[var][hr_idx] += data_dict[var] #np.nansum((comp[var][hr_idx],data_dict[var]))
                    comp[var+"_count"][hr_idx] += 1# comp[var+"_count"][hr_idx] + (~np.isnan(arr)).astype(int)

    for field in list(comp.keys()):
        comp[field]=xr.DataArray(comp[field],coords=[samp_period],dims=["hour"],name=field)

    comp=[comp[field] for field in list(comp.keys())]
    comp=xr.merge(comp)
    return comp


###### PRIMARY HOVMOELLER FUNCTION ######
# Parallelisation done on dates within period. Split period into 10 groups of 4 days. Idx specifies the group.

def parallelise(idx):
    # master dictionary in which composites stored for different variables. Converted to dataset at end.
    # Stores numpy arrays of zeros for variable, count and variable anomalies (if included). Vastly reduces computation times of long sets of means.
    comp={}
    
    for field in field_vars:
        comp[field]=np.zeros((801,len(samp_period)))
        comp[field+"_count"]=np.zeros((801,len(samp_period)))
        if fields in ["hfxs","synoptic"]:
            comp[field+"_anom"]=np.zeros((801,len(samp_period)))
    if fields=="2dfields": 
        for field in ["sm_anom","q925_anom","t925_anom","tcw_anom","ushear650_925_anom","tcc_anom","pblh_anom","theta_e925_anom","pt925_anom"]:
            comp[field]=np.zeros((801,len(samp_period)))

    # load climatology fields to take anomalies from (all Control hourly means)
    if fields=="2dfields":
        sm_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_sm_1p5km.nc").STASH_m01s08i223[:,:2200,:3300]
        tcw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_tcw_1p5km.nc").STASH_m01s30i461[:,:2200,:3300]
        tcc_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_tcc_1p5km.nc").STASH_m01s09i217[:,:2200,:3300]
        pbl_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_pbl_depth_1p5km.nc")[stash_dict["pbl_depth"]][:,:2200,:3300]
        q925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_q925_1p5km.nc").STASH_m01s30i205[:,:2200,:3300]
        t925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_t925_1p5km.nc").STASH_m01s30i204[:,:2200,:3300]
        te925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_theta_e925_1p5km.nc").equivalent_potential_temperature[:,:2200,:3300]
        pt925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_pt925_1p5km.nc").potential_temperature[:,:2200,:3300]
        shear_clim=(xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_u650_1p5km.nc").STASH_m01s30i201[:,:2200,:3300] 
                    - xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_u925_1p5km.nc").STASH_m01s30i201[:,:2200,:3300])
        #v925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_v925_1p5km.nc").STASH_m01s30i202.sel(hour=samp_hr)[:2200,:3300]
        #t2_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_%02dZ_t2_1p5km.nc"%samp_hr).STASH_m01s03i236.isel(hour=0)[:2200,:3300]
        #q2_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_%02dZ_q2_1p5km.nc"%samp_hr).STASH_m01s03i237.sel(hour=samp_hr-1)[:2200,:3300]

        # Since "corePRE" is making reference to the 40 day Control, we now need different climatologies:
        if method=="corePRE" and sim_day==0:
            sm_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_sm_1p5km.nc").STASH_m01s08i223[:,:2200,:3300]
            tcw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_tcw_1p5km.nc").STASH_m01s30i461[:,:2200,:3300]
            tcc_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_tcc_1p5km.nc").STASH_m01s09i217[:,:2200,:3300]
            pbl_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_pbl_depth_1p5km.nc")[stash_dict["pbl_depth"]][:,:2200,:3300]
            q925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_q925_1p5km.nc").STASH_m01s30i205[:,:2200,:3300]
            t925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_t925_1p5km.nc").STASH_m01s30i204[:,:2200,:3300]
            te925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_theta_e925_1p5km.nc").equivalent_potential_temperature[:,:2200,:3300]
            pt925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_pt925_1p5km.nc").potential_temperature[:,:2200,:3300]
            shear_clim=(xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_u650_1p5km.nc").STASH_m01s30i201[:,:2200,:3300] 
                        - xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_u925_1p5km.nc").STASH_m01s30i201[:,:2200,:3300])
    
    elif fields=="hfxs":
        lh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_lhfx_1p5km.nc").STASH_m01s03i234[:,:2200,:3300]
        sh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_shfx_1p5km.nc").STASH_m01s03i217[:,:2200,:3300]
        sw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_sw_nsfc_1p5km.nc").STASH_m01s01i201
        lw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_full_mean_lw_nsfc_1p5km.nc").STASH_m01s02i201
        
        if method=="corePRE" and sim_day==0:
            lh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_lhfx_1p5km.nc").STASH_m01s03i234[:,:2200,:3300]
            ref=load_file(sim,pd.Timestamp("2006-07-27"),12,"surface_vars",stash_dict["lhfx"])[0,:,:]
            sh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_shfx_1p5km.nc").surface_upward_sensible_heat_flux[:,:2200,:3300]
            sh_clim=sh_clim.assign_coords(latitude=ref.latitude)
    
            sw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_sw_nsfc_1p5km.nc").STASH_m01s01i201
            lw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_run_mean_lw_nsfc_1p5km.nc").STASH_m01s02i201
            #ref=load_file(sim,pd.Timestamp("2006-07-27"),12,"surface_vars",stash_dict["sw_nsfc"])[:,:2200,:3300]
            #sw_clim=sw_clim.interp(latitude=ref.latitude,longitude=ref.longitude)
            #lw_clim=lw_clim.interp(latitude=ref.latitude,longitude=ref.longitude)

    
    ################# DAILY FILE SELECTION ######################
    # Segmentation of full period for serial paraellelisation
    period_segment=period[int(4*idx):int(4*(idx+1))]
    
    for j, date in enumerate(period_segment):
        print(date)

        # Have to be careful about selection of right MCSs - to access Day 2 in sensitivity experiment, need Day 1 date + hour > 24; thus need to shift table selection date forwards
        # However we don't want to do that if trying to use fields from one day, and storms from the other!
        date2=date+pd.Timedelta(sim_day,"d")
            
        day_data=MCS_data[(MCS_data["day"]==date2.day) & (MCS_data["month"]==date2.month) & (MCS_data["hour"]==loc_hr)]
        data_dict={}

        for hr_idx, samp_hr in enumerate(samp_period):
            if samp_hr<=0 and date==pd.Timestamp("2006-07-25"):
                continue

            if samp_hr<7 and sim_day==0:
                clim_hr=samp_hr%24
            else:
                clim_hr=samp_hr+sim_day*24
                
            #print(samp_hr)
            if fields=="2dfields":
                data_dict["sm"]=load_file(sim,date,samp_hr+sim_day*24,"soil_moistures",stash_dict["sm"])[0,:,:] 
                data_dict["sm_anom"] = data_dict["sm"] - sm_clim.sel(hour=clim_hr)
                data_dict["tcw"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["tcw"])[0,:,:]
                data_dict["tcw_anom"] = data_dict["tcw"] - tcw_clim.sel(hour=clim_hr)
                data_dict["tcc"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["tcc"])[0,:,:]
                data_dict["tcc_anom"] = data_dict["tcc"] - tcc_clim.sel(hour=clim_hr)
                # NOT converetd into mm/hr
                data_dict["precip"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["precip"])[0,:,:]
                data_dict["pblh"] = load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["pbl_depth"])[0,:,:]
                data_dict["pblh_anom"] = data_dict["pblh"] - pbl_clim.sel(hour=clim_hr)
                t925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["t_plevs"]).sel(PLEVS=925)[0,:,:]
                q925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["q_plevs"]).sel(PLEVS=925)[0,:,:]
                q800=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["q_plevs"]).sel(PLEVS=800)[0,:,:]
                data_dict["t925"]=t925.where(t925!=0)
                data_dict["q925"]=q925.where(q925!=0)
                data_dict["q800"]=q800.where(q800!=0)
                data_dict["t925_anom"]=data_dict["t925"] - t925_clim.sel(hour=clim_hr)
                data_dict["q925_anom"]=data_dict["q925"] - q925_clim.sel(hour=clim_hr)
                u925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["u_plevs"]).sel(PLEVS=925)[0,:,:]
                v925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["v_plevs"]).sel(PLEVS=925)[0,:,:]
                data_dict["ushear650_925"]=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["u_plevs"]).sel(PLEVS=650)[0,:,:] - u925.where(u925!=0)
                data_dict["ushear650_925_anom"]=data_dict["ushear650_925"] - shear_clim.sel(hour=clim_hr)

                # mask out sub-surface plevel values, which are stored as zeros.
                data_dict["u925"]=u925.where(u925!=0)
                data_dict["v925"]=v925.where(v925!=0)
                div=mpcalc.divergence(data_dict["u925"],data_dict["v925"],crs=ccrs.PlateCarree())         
                data_dict["horizontal_divergence"]=div.metpy.dequantify()
                
                pvals=925*np.ones(t925.shape)
                data_dict["pt925"]=mpcalc.potential_temperature(pvals*units("hPa"),data_dict["t925"]).metpy.dequantify()
                data_dict["pt925_anom"]=data_dict["pt925"] - pt925_clim.sel(hour=clim_hr)
                td925=mpcalc.dewpoint_from_specific_humidity(pvals*units("hPa"),t925.where(t925!=0),q925.where(q925!=0))
                data_dict["theta_e925"]=mpcalc.equivalent_potential_temperature(pvals*units("hPa"),t925.where(t925!=0),td925).metpy.dequantify()
                data_dict["theta_e925_anom"]=data_dict["theta_e925"] - te925_clim.sel(hour=clim_hr)
                data_dict["td925"]=td925.metpy.dequantify()

                # CAPE proxy is different between 925hPa theta-e and 650hPa fully saturated theta-e
                t650=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["t_plevs"]).sel(PLEVS=650)[0,:,:]
                tds=mpcalc.dewpoint_from_relative_humidity(t650,100/925 * pvals *units('percent'))
                capeprox=mpcalc.equivalent_potential_temperature(650/925 * pvals * units("hPa"),t650,tds).metpy.dequantify()
                data_dict["capeprox"] =  data_dict["theta_e925"] - capeprox

                
            elif fields=="hfxs":
                data_dict["lhfx"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["lhfx"])[0,:,:]
                if sim=="control":
                    data_dict["shfx"]=load_file(sim,date,samp_hr+sim_day*24,"sensible_hfx_control",stash_dict["shfx"])[0,:,:].interp(latitude=data_dict["latent_heat_flux"].latitude)
                else:
                    data_dict["shfx"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["shfx"])[0,:,:].interp(latitude=data_dict["lhfx"].latitude)
                data_dict["ae"]=data_dict["shfx"]+data_dict["lhfx"]
                data_dict["ef"]=data_dict["lhfx"]/data_dict["ae"]
                                
                data_dict["shfx_anom"] = data_dict["shfx"] - sh_clim.sel(hour=clim_hr)
                data_dict["lhfx_anom"] = data_dict["lhfx"] - lh_clim.sel(hour=clim_hr)
                data_dict["ae_anom"] = data_dict["shfx_anom"] + data_dict["lhfx_anom"]
                data_dict["ef_anom"] = data_dict["ef"] - (lh_clim/(sh_clim+lh_clim)).sel(hour=clim_hr)

                data_dict["sw_nsfc"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["sw_nsfc"])[0,:,:]
                data_dict["lw_nsfc"]=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["lw_nsfc"])[0,:,:]
                data_dict["sw_nsfc_anom"] = data_dict["sw_nsfc"] - sw_clim.sel(hour=clim_hr) 
                data_dict["lw_nsfc_anom"] = data_dict["lw_nsfc"] - lw_clim.sel(hour=clim_hr) 
            
    
            elif fields=="moisture_fluxes":                
                q=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["q_plevs"])[0,1:lid+1,:,:]
                u=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["u_plevs"])[0,1:lid+1,:,:]
                v=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["v_plevs"])[0,1:lid+1,:,:]

                # Calculate moisture fluxes and their divergence
                qu=q*u#q.where(q!=0)*u.where(u!=0) - ideally would exclude sub-surface zeroes, but nans destroy the integrals and error is very small
                qv=q*v#.where(q!=0)*v.where(v!=0)
                MFD=mpcalc.divergence(qu,qv,crs=ccrs.PlateCarree())
                MFD=MFD.metpy.dequantify()
                print("divergence") # its slow - useful waymarker
                
                data_dict["MF_u925"]=qu.where(qu!=0).sel(PLEVS=925)
                data_dict["MF_v925"]=qv.where(qv!=0).sel(PLEVS=925)
                data_dict["MF_u"]=qu # note NOT integrated - do later on composites. Much quicker than integrating entire field
                data_dict["MF_v"]=qv
                data_dict["MFD_925"]=MFD.where(MFD!=0).sel(PLEVS=925)                    
                data_dict["MFD"]=MFD


            ################### SAMPLING ROUTINE ###############################
            for core_loc in day_data[[loc_y,loc_x]].values:
                for var in field_vars:
                    # extract numpy array of values about each sample point (lat, lon). data_dict entry is full field
                    arr=composite_section(data_dict[var], core_loc[0], core_loc[1])
                    # Do column integrals at composite level and without xArray wrappers
                    if var in ["MFD","MF_u","MF_v"]:
                        arr=np.trapz(arr,x=q.PLEVS.values,axis=0)
                    # update same-sized array stored in comp. Starts as zeros. nansum important to account for sub-surface pressure levels etc
                    comp[var][:,hr_idx] = np.nansum((comp[var][:,hr_idx],arr), axis=0)
                    # we ultimately want to compute a mean, so also track sample size. Needs to be grid-level to account for spatial patterns of nans.
                    comp[var+"_count"][:,hr_idx] = comp[var+"_count"][:,hr_idx] + (~np.isnan(arr)).astype(int)

                    # for most anomalies, the key is in comp only. Thus treat separately; does therefore cause an issue if ever have var and var+"_anom" in field_vars
                    if var+"_anom" in list(comp.keys()):
                        arr=composite_section(data_dict[var+"_anom"], core_loc[0], core_loc[1])
                        comp[var+"_anom"][:,hr_idx] = np.nansum((comp[var+"_anom"][:,hr_idx],arr), axis=0)
            
            ########################################################################

    ##### FINALISE COMPOSITE #####
    # Once all days in period_segment have been sampled, convert arrays in comp to dataArrays
    # Latitudinal coordinates applied as kilometres as from centre, from -600 to 600
    for field in list(comp.keys()):
        comp[field]=xr.DataArray(comp[field],coords=[np.arange(-600,601,1.5),samp_period],
                            dims=["latitude","hour"],name=field)

    # Convert comp dictionary (now one dataArray per variable + its count + its anom) into a single dataset
    comp=[comp[field] for field in list(comp.keys())]
    comp=xr.merge(comp)
    # Output - a dataset of composites for each period_segment
    return comp

################## ACTIVATE PARALLELISE ###########################
# psize is set around argparse input. Default 10, 4 if using args.split
p = Pool(psize)
print(psize)
if fields=="instability":
    part_composites=p.map(instability_evolution,np.arange(psize))
    out=xr.concat(part_composites,dim="part")
else:
    part_composites=p.map(parallelise,np.arange(psize))
    out=xr.concat(part_composites,dim="part")
print(out.data_vars)

count=0
################## CLEAN UP OUTPUTS ############################

# Core clean-up. Combine the collated sums and counts; combine to take mean; include anomalies and probabilities
if args.split is None:
    for field in field_vars:
        N=out[field+"_count"].sum(dim="part").values
        out[field]=out[field].sum(dim="part")/N
        count=max(count,np.max(N))
        out=out.drop_vars([field+"_count"])
    
        if field+"_anom" in list(out.data_vars):
            out[field+"_anom"]=out[field+"_anom"].sum(dim="part")/N
            
# If split routine used, do everything except take averages. This allows for accurate combination of the splits when all files output
else:
    for field in field_vars:
        #print(field,out[field])
        out[field+"_count"]=out[field+"_count"].sum(dim="part")
        out[field]=out[field].sum(dim="part")
    
        if field+"_anom" in list(out.data_vars):
            out[field+"_anom"]=out[field+"_anom"].sum(dim="part")

out.attrs["number_cores"]=count


if "diffPmax" in method:
    start_str=sim2+"_%02dzSM"%loc_hr
else:
    start_str=sim2+"_%02dzMCS"%loc_hr

# If arg.split has been used, you get a separate .nc file per split
if args.split is None:
    out.to_netcdf("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/composites/Control48_climatology/{}_{}_filt_{}_hmoeller.nc".format(start_str,fields,method))
else:
    out.to_netcdf("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/composites/Control48_climatology/{}_{}_flt_{}_hmoeller-P{}.nc".format(start_str,fields,method,args.split))
    