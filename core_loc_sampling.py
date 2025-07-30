from sm_scales_utils import *

### ! WARNING: THIS SCRIPT IS DESIGNED TO RUN ON MACHINES WHICH PERMIT SERIAL PARALLEL PROCESSING ! ###

###################################################################################
# - sim : ["control","sens"] - for continuous 40 day Control, or ANY set of 48hr simulations, respectively. All paper results use Sens.
# - expt : if sens, sensitivity experiment to select. Options: ["control_48hr_runs", "wg_mcs", "large_only"]
# - day : if sens, day of experiment to sample (typically wish to split). Options 0 or 1, i.e. pythonic indexing.
# - method : specification of the type of sampling point being composited. Options ["core","diffPmax","diffPmaxWET"] (last two being "MesoDRY" and "MesoWET" SM patches, first MCS cores)
# - loc_time : the hour at which sampling points have been identified, UTC. Default 17 for "core", 9 for SM patches
# - anoms : whether to sample anomalous or absolute fields
# - update : use to update an existing table. Manually copy out bits you don't want to do again!
###################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=True) 
parser.add_argument("-e", "--expt", required=False, default="control_48hr_runs")
parser.add_argument("-d", "--day", required=False, type=int, default=0)
parser.add_argument("-m", "--method", required=False, default="core")
parser.add_argument("-lt", "--loc_time", required=False, default=17, type=int)
parser.add_argument("-st", "--samp_time", required=False, default=12, type=int)
parser.add_argument("-a", "--anoms", required=False)
parser.add_argument("-u", "--update", required=False)
args = parser.parse_args()

sim=args.sim
expt=args.expt
sim_day=args.day
if "control" in sim.lower():
    sim2="Control_run"
else:
    period=period[:-2]
    sim2=expt.capitalize()+"_D%s"%(sim_day+1)

# Set afact parameter, which determines if anomalies calculated
if args.anoms is None:
    astr,afact="",0
else:
    astr,afact="_anoms",1

method=args.method
loc_hr=args.loc_time
hrs=[loc_hr]
samp_hr=args.samp_time


if expt!="control_48hr_runs":
    expt="sensitivity_runs_"+expt

if method=="diffPmax":
    if args.update is None:
        MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/Wg_mcs_D1_sm-diff_{:02d}Z_pmax_locs.csv".format(loc_hr))
    else:
        print("Updating table")
        MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/{}_{}_sm-diff_{:02d}Z_pmax_envfields{}.csv".format(sim.capitalize(),expt,loc_hr,astr))
    loc_x,loc_y="pmax_lon","pmax_lat"
    groupt="date"
    
else:
    groupt="time"
    if loc_hr==17:
        if args.update is None:
            MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{:02d}Z_Sahel_filt_cores.csv".format(sim2,loc_hr))
        else:
            print("Updating table")
            MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{:02d}Z_Sahel_filt_core_envfields{}.csv".format(sim2,loc_hr,astr))
            print("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{:02d}Z_Sahel_filt_core_envfields{}.csv".format(sim2,loc_hr,astr))
        loc_x,loc_y="core_lon","core_lat"
    else:
        MCS_data=pd.read_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_comphours.csv".format(sim2))
        full_tracks=glob.glob("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_tracks_rain.csv".format(sim2))
        full_tracks["start_time"]=pd.to_datetime(full_tracks["start_time"])

    if sim=="sens":
        MCS_data=MCS_data[MCS_data.sim_day==sim_day]

        
#RESTRICT COMPOSITES TO SAHEL
MCS_data=MCS_data[(MCS_data[loc_y]>9) & (MCS_data[loc_y]<=19) & (MCS_data[loc_x]>=348) & (MCS_data[loc_x]<=378)]

# Load climatology fields for anomaly calculation
sm_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_sm_1p5km.nc"%(sim_day+1)).STASH_m01s08i223[:,:2200,:3300]
tcw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_tcw_1p5km.nc"%(sim_day+1)).STASH_m01s30i461.sel(hour=samp_hr)[:2200,:3300]
q925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_q925_1p5km.nc"%(sim_day+1)).STASH_m01s30i205.sel(hour=samp_hr)[:2200,:3300]
v925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_v925_1p5km.nc"%(sim_day+1)).STASH_m01s30i202.sel(hour=samp_hr)[:2200,:3300]
t925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_t925_1p5km.nc"%(sim_day+1)).STASH_m01s30i204.sel(hour=samp_hr)[:2200,:3300]
theta_e925_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_theta_e925_1p5km.nc"%(sim_day+1)).equivalent_potential_temperature.sel(hour=samp_hr)[:2200,:3300]
t2_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_t2_1p5km.nc"%(sim_day+1)).STASH_m01s03i236.isel(hour=0)[:2200,:3300]
q2_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_q2_1p5km.nc"%(sim_day+1)).STASH_m01s03i237.sel(hour=samp_hr-1)[:2200,:3300]
precip_clim=3600*xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_precip_1p5km.nc"%(sim_day+1)).STASH_m01s04i203[:,:2200,:3300]
tcc_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_tcc_1p5km.nc"%(sim_day+1)).STASH_m01s09i217.sel(hour=samp_hr)[:2200,:3300]
lh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_lhfx_1p5km.nc"%(sim_day+1)).STASH_m01s03i234[:,:2200,:3300]
#ref_sh=load_file(sim,pd.Timestamp("2006-07-27"),samp_hr+sim_day*24,"sensible_hfx_control",stash_dict["shfx"])[0,:,:]
sh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_shfx_1p5km.nc"%(sim_day+1)).STASH_m01s03i217[:,:2200,:3300]
#sh_clim=sh_clim.assign_coords(latitude=ref_sh.latitude)
sw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_sw_nsfc_1p5km.nc"%(sim_day+1)).STASH_m01s01i201[:,:2200,:3300]
lw_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_lw_nsfc_1p5km.nc"%(sim_day+1)).STASH_m01s02i201[:,:2200,:3300]
pblh_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_pbl_depth_1p5km.nc"%(sim_day+1)).STASH_m01s00i025[:,:2200,:3300]
shear_clim=xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_u650_1p5km.nc"%(sim_day+1)).STASH_m01s30i201[:,:2200,:3300
                    ] - xr.open_dataset("/gws/nopw/j04/lmcs/bmaybee/lmcs_run_outputs/mean_states/Control_48hr_runs_D%s_mean_u925_1p5km.nc"%(sim_day+1)).STASH_m01s30i201[:,:2200,:3300]


###### ALL ROUTINES NESTED WITHIN THIS FUNCTION ######
# Parallelisation done on dates within period. Split period into 10 groups of 4 days. Idx specifies the group.

def parallelise(idx):
    # Get mean/max of variable over 1 deg box
    def sample_mean_var(da,lat,lon):
        da=da.sel(latitude=slice(lat-0.5,lat+0.5),longitude=slice(lon-0.5,lon+0.5))
        return float(da.mean())
    def sample_max_var(da,lat,lon):
        da=da.sel(latitude=slice(lat-0.5,lat+0.5),longitude=slice(lon-0.5,lon+0.5))
        return float(da.max())

    # Slightly experimental function to get properties of AEJ and ITD from latitudinal search within 4 degree longitudinal slice about sample point.
    # Output not used in paper
    def get_synoptics(row):
        aej_loc=aej.sel(longitude=slice(row[loc_y]-2,row[loc_x]+2)).mean(dim="longitude").values
        row["mcs_aej_max"]=np.nanmin(aej_loc)
        row["mcs_aej_lat"]=aej.latitude.values[np.unravel_index(np.argmin(aej_loc),aej_loc.shape)[0]]

        row["mcs_v925_min"]=np.nanmin(v925.sel(latitude=slice(row[loc_y]-2,row[loc_y]+2),longitude=slice(row[loc_x]-2,row[loc_x]+2)).values)
        itd_loc=itd_v.sel(longitude=slice(row[loc_x]-2,row[loc_x]+2)).mean(dim="longitude")
        try:
            row["mcs_itd_v925_lat"]=np.nanmin(itd_loc.latitude.values[np.sign(itd_loc)<=0])
        except:
            row["mcs_itd_v925_lat"]=np.NaN

        row["mcs_td925_min"]=np.nanmin(td925.sel(latitude=slice(row[loc_y]-2,row[loc_y]+2),longitude=slice(row[loc_x]-2,row[loc_x]+2)).values)
        itd_loc=itd_td.sel(longitude=slice(row[loc_x]-2,row[loc_x]+2)).mean(dim="longitude")
        try:
            row["mcs_itd_td925_lat"]=np.nanmin(itd_loc.latitude.values[np.sign(itd_loc-13)<=0])
        except:
            row["mcs_itd_td925_lat"]=np.NaN
        
        return row

    # Function to get thermodynamic instability measures from mean 1 degree soundings on pressure levels. Cape_prof=True outputs all non-zero CAPE outputs
    def get_soundings(row,cape_prof=False):
        prof_t=t.sel(latitude=slice(row[loc_y]-0.5,row[loc_y]+0.5),longitude=slice(row[loc_x]-0.5,row[loc_x]+0.5)).mean(dim=["latitude","longitude"])
        prof_t=prof_t.dropna(dim="PLEVS")

        prof_q=q.sel(latitude=slice(row[loc_y]-0.5,row[loc_y]+0.5),longitude=slice(row[loc_x]-0.5,row[loc_x]+0.5)).mean(dim=["latitude","longitude"])
        prof_q=prof_q.dropna(dim="PLEVS")
        prof_p=prof_q.PLEVS

        prof_td=mpcalc.dewpoint_from_specific_humidity(prof_p*units("hPa"),prof_t*units("K"),prof_q*units("kg/kg"))
        lcl=mpcalc.lcl(prof_p[0]*units("hPa"),prof_t[0]*units("K"),prof_td[0])
        cape_cin=np.zeros((len(prof_p),2))

        # Get instability for lowest-level parcel
        try:
            parc=mpcalc.parcel_profile(prof_p*units("hPa"),prof_t[0]*units("K"),prof_td[0])
            cpe=mpcalc.cape_cin(prof_p*units("hPa"),prof_t*units("K"),prof_td,parc)
            cape_cin[0,0]=cpe[0].magnitude
            cape_cin[0,1]=cpe[1].magnitude
            row["CAPE"]=cpe[0].magnitude
            row["CIN"]=cpe[1].magnitude
            row["LCL"]=lcl[0].magnitude
        except:
            print("Failed")
            print(prof_p,prof_t[0],prof_td[0])
            row["CAPE"],row["CIN"],row["LCL"],row["ICAPE"],row["ICIN"]=-np.NaN,np.NaN,np.NaN,np.NaN,np.NaN
            return row

        # Calculate CAPE/CIN for all values below the LCL (i.e. where non-zero)
        for i, p in enumerate(prof_p.values[1:]):
            if prof_p.sel(PLEVS=p) > lcl[0].magnitude and prof_t.sel(PLEVS=p) > lcl[1].magnitude:
                parc=mpcalc.parcel_profile(prof_p.sel(PLEVS=slice(p,None))*units("hPa"),
                                           prof_t.sel(PLEVS=p)*units("K"),
                                           prof_td.sel(PLEVS=p))
                cpe=mpcalc.cape_cin(prof_p.sel(PLEVS=slice(p,None))*units("hPa"),
                                    prof_t.sel(PLEVS=slice(p,None)).sel(PLEVS=slice(p,None))*units("K"),
                                    prof_td.sel(PLEVS=slice(p,None)),
                                    parc)
                cape_cin[i,0]=cpe[0].magnitude
                cape_cin[i,1]=cpe[1].magnitude
            else:
                continue

        # to get ICAPE, integrate over those values (ICIN less well defined); sign for limits order.
        row["ICAPE"]=-1/9.81 * np.trapz(cape_cin[:,0],x=100*prof_p.values)
        row["ICIN"]=-1/9.81 * np.trapz(cape_cin[:,1],x=100*prof_p.values)
        # output all the intermediate plevel CAPE values
        if cape_prof:
            for i in range(len(cape_cin)):
                if cape_cin[i,0]!=0:
                    row["CAPE_"+str(prof_p.values[i])]=cape_cin[i,0]
            print(row.filter(regex="CAPE_"))
        return row
    
    period_segment=period[int(4*idx):int(4*(idx+1))]# covers 4 day segments; Pool=10
    update=[]
    
    for j, date in enumerate(period_segment):
        print(date)
        # Have to be careful about selection of right MCSs - to access Day 2 in sensitivity experiment, need Day 1 date + hour > 24; thus need to shift table selection date forwards
        date2=date+pd.Timedelta(sim_day,"d") 
        date_storms=MCS_data[(MCS_data["day"]==date2.day) & (MCS_data["month"]==date2.month) & (MCS_data["hour"].isin(hrs))]
        
        sm9=load_file(sim,date,9+sim_day*24,"soil_moistures",stash_dict["sm"])
        sm12=load_file(sim,date,12+sim_day*24,"soil_moistures",stash_dict["sm"])

        # Load PBL fields
        u650=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["u_plevs"],region="full").sel(PLEVS=650)[0,:,:]
        u925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["u_plevs"],region="full").sel(PLEVS=925)[0,:,:]
        v925=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["v_plevs"],region="full").sel(PLEVS=925)[0,:,:]
        t=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["t_plevs"],region="full")[0,1:,:,:]
        q=load_file(sim,date,samp_hr+sim_day*24,"model-diagnostics",stash_dict["q_plevs"],region="full")[0,1:,:,:]
        # Exclude subsurface values
        t=t.where(t!=0)
        q=q.where(q!=0)
        v925=v925.where(v925!=0)
        u925=u925.where(u925!=0)
        t2=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["t2"])[0,:,:]
        tcw=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["tcw"])[0,:,:]
        precip=3600*load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["precip"])[0,:,:]
        q2=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["q2"])[0,:,:]
        tcc=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["tcc"])[0,:,:]
        pblh=load_file(sim,date,samp_hr+sim_day*24,"surface_vars",stash_dict["pbl_depth"])[0,:,:]
        ushear=u650-u925
        div=mpcalc.divergence(u925,v925,crs=ccrs.PlateCarree())         
        div=div.metpy.dequantify()
        
        pvals=925*np.ones(t.sel(PLEVS=925).shape)
        td925=mpcalc.dewpoint_from_specific_humidity(pvals*units("hPa"),t.sel(PLEVS=925),q.sel(PLEVS=925))
        theta_e925=mpcalc.equivalent_potential_temperature(pvals*units("hPa"),t.sel(PLEVS=925),td925).metpy.dequantify()
        td925=td925.metpy.dequantify()

        tds=mpcalc.dewpoint_from_relative_humidity(t.sel(PLEVS=650),100/925 * pvals *units('percent'))
        capeprox=mpcalc.equivalent_potential_temperature(650/925 * pvals * units("hPa"),t.sel(PLEVS=650),tds).metpy.dequantify()
        capeprox = theta_e925 - capeprox
        
        aej=u650.sel(latitude=slice(8,20),longitude=slice(347,379)) #.coarsen(latitude=10,longitude=10,boundary="trim").mean()
        itd_v=v925#.coarsen(latitude=10,longitude=10,boundary="trim").mean()
        itd_td=td925#.coarsen(latitude=10,longitude=10,boundary="trim").mean()
        
        # morning mean heat and radiative fluxes
        flux_dict={}
        if sim=="control":
            sh_accum=load_file(sim,date,samp_hr-3,"sensible_hfx_control",stash_dict["shfx"])[0,:,:]
        else:
            sh_accum=load_file(sim,date,samp_hr-3+sim_day*24,"surface_vars",stash_dict["shfx"])[0,:,:]
        lh_accum=load_file(sim,date,samp_hr-3+sim_day*24,"surface_vars",stash_dict["lhfx"])[0,:,:]
        sw_accum=load_file(sim,date,samp_hr-3+sim_day*24,"surface_vars",stash_dict["sw_nsfc"])[0,:,:]
        lw_accum=load_file(sim,date,samp_hr-3+sim_day*24,"surface_vars",stash_dict["lw_nsfc"])[0,:,:]
        for hr in np.arange(samp_hr-3,samp_hr+1):
            if sim=="control":
                sh=load_file(sim,date,hr,"sensible_hfx_control",stash_dict["shfx"])[0,:,:] 
            else:
                sh=load_file(sim,date,hr+sim_day*24,"surface_vars",stash_dict["shfx"])[0,:,:] 
            lh=load_file(sim,date,hr+sim_day*24,"surface_vars",stash_dict["lhfx"])[0,:,:] 
            sw=load_file(sim,date,hr+sim_day*24,"surface_vars",stash_dict["sw_nsfc"])[0,:,:]
            lw=load_file(sim,date,hr+sim_day*24,"surface_vars",stash_dict["lw_nsfc"])[0,:,:]
            sh_accum=sh + sh_accum
            lh_accum=lh + lh_accum
            sw_accum=sw + sw_accum
            lw_accum=lw + lw_accum

        flux_dict["sh"]=sh - afact*sh_clim.sel(hour=samp_hr)
        flux_dict["lh"]=lh - afact*lh_clim.sel(hour=samp_hr)
        flux_dict["ae"]=flux_dict["sh"]+flux_dict["lh"]
        flux_dict["ef"]=flux_dict["lh"]/(flux_dict["sh"]+flux_dict["lh"])
        flux_dict["sw"]=sw - afact*sw_clim.sel(hour=samp_hr)
        flux_dict["lw"]=lw - afact*lw_clim.sel(hour=samp_hr)
        
        flux_dict["sh_accum"]=sh_accum - afact*sh_clim.sel(hour=slice(samp_hr-3,samp_hr)).sum(dim="hour")
        flux_dict["lh_accum"]=lh_accum - afact*lh_clim.sel(hour=slice(samp_hr-3,samp_hr)).sum(dim="hour")
        flux_dict["sw_accum"]=sw_accum - afact*sw_clim.sel(hour=slice(samp_hr-3,samp_hr)).sum(dim="hour")
        flux_dict["lw_accum"]=lw_accum - afact*lw_clim.sel(hour=slice(samp_hr-3,samp_hr)).sum(dim="hour")
        flux_dict["ae_accum"]=flux_dict["lh_accum"]+flux_dict["sh_accum"]
        flux_dict["ef_accum"]=flux_dict["lh_accum"]/flux_dict["ae_accum"]
        
        
        
        # Now start to populate table - remember afact is anomaly parameter
        date_storms["09Z_sm"]=date_storms.apply(lambda x: 
                                        sample_mean_var(sm9-afact*sm_clim.sel(hour=samp_hr-3),x[loc_y],x[loc_x]),axis=1)
        date_storms["12Z_sm"]=date_storms.apply(lambda x: 
                                        sample_mean_var(sm12-afact*sm_clim.sel(hour=samp_hr),x[loc_y],x[loc_x]),axis=1)
        date_storms["t2"]=date_storms.apply(lambda x: sample_mean_var(t2-afact*t2_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["q2"]=date_storms.apply(lambda x: sample_mean_var(q2-afact*q2_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["t925"]=date_storms.apply(lambda x: sample_mean_var(t.sel(PLEVS=925)-afact*t925_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["q925"]=date_storms.apply(lambda x: sample_mean_var(q.sel(PLEVS=925)-afact*q925_clim,x[loc_y],x[loc_x]),axis=1)
        # no climatology available.
        date_storms["td925"]=date_storms.apply(lambda x: sample_mean_var(td925,x[loc_y],x[loc_x]),axis=1)
        date_storms["theta_e925"]=date_storms.apply(lambda x: sample_mean_var(theta_e925-afact*theta_e925_clim,x[loc_y],x[loc_x]),axis=1)
        # no climatology available.
        date_storms["CAPE-proxy"]=date_storms.apply(lambda x: sample_mean_var(capeprox,x[loc_y],x[loc_x]),axis=1)
        # no climatology available (also usually near 0).
        date_storms["div925"]=date_storms.apply(lambda x: sample_mean_var(div,x[loc_y],x[loc_x]),axis=1)
        date_storms["tcw"]=date_storms.apply(lambda x: sample_mean_var(tcw-afact*tcw_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["rain_mean"]=date_storms.apply(lambda x: sample_mean_var(precip-afact*precip_clim,x[loc_y],x[loc_x]),axis=1)
        try:
            date_storms["rain_max"]=date_storms.apply(lambda x: sample_max_var(precip-afact*precip_clim,x[loc_y],x[loc_x]),axis=1)
        except:
            try:
                print(date_storms.apply(lambda x: sample_max_var(precip-afact*precip_clim,x[loc_y],x[loc_x]),axis=1))
            except:
                print(date, "rain max failure")
        date_storms["tcc"]=date_storms.apply(lambda x: sample_mean_var(tcc-afact*tcc_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["pblh"]=date_storms.apply(lambda x: sample_mean_var(pblh-afact*pblh_clim,x[loc_y],x[loc_x]),axis=1)
        date_storms["ushear650_925"]=date_storms.apply(lambda x: sample_mean_var(ushear-afact*shear_clim,x[loc_y],x[loc_x]),axis=1)
        
        for var in ["sh","lh","ae","ef","sw","lw"]:#
            date_storms[var]=date_storms.apply(lambda x: sample_mean_var(flux_dict[var],x[loc_y],x[loc_x]),axis=1)
            date_storms[var+"_accum"]=date_storms.apply(lambda x: sample_mean_var(flux_dict[var+"_accum"],x[loc_y],x[loc_x]),axis=1)
            
        date_storms=date_storms.apply(get_synoptics,axis=1)
        date_storms=date_storms.apply(get_soundings,axis=1)
        date_storms=date_storms.apply(get_soundings,cape_prof=True,axis=1)
    
        update.append(date_storms)

    update=pd.concat(update,axis=0)
    return update

psize=10
p = Pool(psize)
part_composites=p.map(parallelise,np.arange(psize))
out=pd.concat(part_composites,axis=0)
out=out.sort_values(by=groupt) # "date" or "time" - foolishly different column names for different methods!

if method=="diffPmax":
    out.to_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/{}_{}_sm-diff_{:02d}Z_pmax_envfields{}.csv".format(sim.capitalize(),expt,loc_hr,astr),index=False)
else:
    out.to_csv("~/LMCS/LMCS_Wafrica_sim/MCS_analysis/Tables/{}_MCS_{:02d}Z_Sahel_filt_{}_envfields{}.csv".format(sim2,loc_hr,method,astr),index=False)
