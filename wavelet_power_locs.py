from sm_scales_utils import *
import wclass
import constants as cnst
import warnings
from scipy.ndimage import label
from itertools import repeat
warnings.filterwarnings("ignore")

wObj = wclass.landwav('SM1p5km_control')
scales = wObj.scales

### ! WARNING: THIS SCRIPT IS DESIGNED TO RUN ON MACHINES WHICH PERMIT SERIAL PARALLEL PROCESSING ! ###

###################################################################################
# - expt : Options ["wg_mcs","large_only"] - choice of of which sensitivity experiment to look at Control difference from. Paper uses wg_mcs, results very similar.
# - idx: Index for group of 3 within the 40 day period - set 0-13
# - day : if sens, day of experiment to sample (typically wish to split). Options 0 or 1, i.e. pythonic indexing.
# - hour : hour at which to compare field spectra. Default (used in paper) 09 UTC
# - field : field to use. SM default and most sensible as that's the sensitivity experiment set up!
###################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--expt")
parser.add_argument("-i", "--idx", type=int)
parser.add_argument("-d", "--day", required=False, type=int, default=0)
parser.add_argument("-hr", "--hour", required=False, type=int, default=9)
parser.add_argument("-f", "--field", required=False, default="sm")
args = parser.parse_args()

expt=args.expt
hr=args.hour
sim_day=args.day
field=args.field

idx=args.idx
period_full=pd.date_range("2006-07-25","2006-09-01")
period=period_full[idx*3:(idx+1)*3]
sim2=expt.capitalize()+"_D%s"%(sim_day+1)

if field=="sm":
    file_list="soil_moistures"
else:
    file_list="surface_vars"

refs=load_file("control",pd.Timestamp("2006-07-25"),9,file_list,stash_dict[field])[0,:,:]
refs=refs.rename({refs.dims[-1]:"longitude",refs.dims[-2]:"latitude"})

# Wrapper for all functionality. Parallelise on single dates and whether to look for positive or negative diffs                
def parallelise(idx,sgn):
    date=period[idx]
    print(date)
    sm_con=load_file("control",date,hr+sim_day*24,file_list,stash_dict[field])[0,:,:]
    sm_sens=load_file("sens",date,hr+sim_day*24,file_list,stash_dict[field])[0,:,:]

    # Get difference field
    diff=sm_sens-sm_con
    diff_sgn=np.sign(diff)
    # Choose which sign to get. For SM, +ve = dry patch (more SM in sens), -ve = wet
    diff=abs(diff.where(diff_sgn==sgn).fillna(0))

    # Wavelet transform using method described in Klein et al, JGR-A (2018)
    wObj = wclass.landwav('SM1p5km_control')
    scales = wObj.scales
    wObj.read_img(diff.values, diff.longitude.values, diff.latitude.values)
    # Different to other transforms, here normalise by the standard deviation of the wavelet transform. Enables signifiance estimate.
    coeffs, power, scales, mults = wObj.applyWavelet(normed='stddev', ge_thresh=None, le_thresh=0)
    print(idx,"filtered")        
    
    out=[]
    for scale in np.arange(21,29): 
        # Currently scale choice: 152 - 640km inclusive
        # Original scale choice: 107 - 640km inclusive. Was too skewed to small patches rather than mesoscale (wavelet scale cutoffs are not sharp!)
        pscale=power[scale,:,:]
        # Identify significant regions, or "hotspots"
        pscale=np.where(pscale>=2,pscale,0) ### KEY CUTOFF - on stddev norm, think = 2sigma value. Previously used np.sqrt(scale)/2 - again, skewed to smaller scales ###
        hotspots=label(pscale)[0]

        # For each region, get power maxmima and centroid location. First label item is the background field values
        out_vals=np.ones((len(np.unique(hotspots))-1,4))
        out_vals[:,0]=scales[scale]*out_vals[:,0]        
        
        for i, lab in enumerate(np.unique(hotspots)[1:]):
            out_vals[i,1]=np.max(pscale[hotspots==lab])
            max_pos=np.unravel_index(np.argmax(np.where(hotspots==lab,pscale,0)), pscale.shape)
            out_vals[i,2]=max_pos[0]
            out_vals[i,3]=max_pos[1]
        out.append(out_vals)

    st=time.time()
    # to isolate unique sample points where have maxima at multiple scales, coarsen to largest sample scale (grid centred on points, hence /2) 
    # Note originally coarsened on smallest scales and grouped by power maxima value - but this all again skews to smallest scales. Current set up prioritises larger scales.
    div=scales[28]/1.5/2
    outdf=pd.DataFrame(np.vstack(out),columns=["scale","pmax","pmax_lat","pmax_lon"])
    outdf["pos"]=np.round(outdf.pmax_lon/div)+np.round(outdf.pmax_lat/div)*np.round(outdf.pmax_lon/div)
    # reduce down to scale with the max power values
    outdf=outdf[outdf.groupby(['pos'])['scale'].transform(max) == outdf['scale']]
    
    outdf.insert(0,"date",date.replace(hour=hr))
    outdf=outdf.drop(columns="pos")

    return outdf

# Start with MesoDRY locations. Starmap needed for parallelise to pick up sign variable!
sgn=1
p=Pool(3)
drylocs=p.starmap(parallelise,zip(np.arange(3),repeat(sgn)))
print("Sign 1 complete")
drylocs=pd.concat(drylocs,axis=0)
drylocs["diff_sign"]=sgn
# TEMPORARY SAVE - overwritten at end
drylocs.to_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/{}_{}-diff_{:02d}Z_pmax_locs-slice{}.csv".format(sim2,field,hr,idx),index=False)

# Then MesoWET
print("Repeating for opposite sign")
sgn=-1
p=Pool(3)
wetlocs=p.starmap(parallelise,zip(np.arange(3),repeat(sgn)))
print("Sign -1 complete")
wetlocs=pd.concat(wetlocs,axis=0)
wetlocs["diff_sign"]=sgn

# Now that worked, combine the two together and overwrite the previous dump.
tab=pd.concat([drylocs,wetlocs])
tab=tab.sort_values(by=["date","scale"])
tab.insert(1,"month",tab.date.dt.month)
tab.insert(2,"day",tab.date.dt.day)
tab.insert(3,"hour",tab.date.dt.hour)
# Convert patch centroid indices to lat/lon coordinates
tab["pmax_lat"]=tab["pmax_lat"].apply(lambda x: float(refs.latitude.values[int(x)]) )
tab["pmax_lon"]=tab["pmax_lon"].apply(lambda x: float(refs.longitude.values[int(x)]) )
tab.to_csv("~/LMCS/LMCS_Wafrica_sim/field_scales/diffPmax_tables/{}_{}-diff_{:02d}Z_pmax_locs-slice{}.csv".format(sim2,field,hr,idx),index=False)