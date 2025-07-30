from sm_scales_utils import *
import wclass
import constants as cnst
import warnings
import pickle
warnings.filterwarnings("ignore")

### ! WARNING: THIS SCRIPT IS DESIGNED TO RUN ON MACHINES WHICH PERMIT SERIAL PARALLEL PROCESSING ! ###

wObj = wclass.landwav('SM1p5km_control')
scales = wObj.scales

##################################################################################
# - sim : ["control","control_48hr_runs","sens"] - for continuous 40 day Control, or 48hr Control simulations, or 48hr sens expts respectively. DIFFERENT TO MOST OTHER PARSERS
# - expt : if sens, sensitivity experiment to select. Options: ["wg_mcs", "large_only"]
# - hour : integer hour at which to analyse spectrum. Defined relative to 00UTC - so 48hr expts first fields are @ hr=7
# - field : if None, get default set ["sm","lhfx","shfx","t2","q2","lw_nsfc","sw_nsfc","tcc","toaolr","toaswr","lcc"]. 
#           Or specify variable code. CODES REFER TO STASH_DICT, MUST BE THERE!
# - portion : index for splitting up period to facilitate parallelisation. Combine output tables once all complete.
#    - field=None -> index within 4 day chunks, i.e. 0-9.
#    - otherwise -> index within 16 day chunks, i.e. 0-2
###################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sim", required=True) #control, control_48hr_runs or sens
parser.add_argument("-e", "--expt", required=False, default="wg_mcs") # wg_mcs or large_only
parser.add_argument("-hr", "--hour", required=False, type=int)
parser.add_argument("-f","--field",required=False)
parser.add_argument("-p", "--portion", required=False, type=int)
args = parser.parse_args()

sim=args.sim
expt=args.expt
hour=args.hour
portion=args.portion
field=args.field

if "control" in sim:
    expt=""


# Apply wavelet transformation, using method and code described in Klein et al, JGR-A (2018)
def wavelet_transform(data,le_thresh=None):
    wObj = wclass.landwav('SM1p5km_control')
    wObj.read_img(data.values, data.longitude.values, data.latitude.values)
    if le_thresh==None:
        coeffs, power, scales, period = wObj.applyWavelet(normed='none')
    else:
        coeffs, power, scales, period = wObj.applyWavelet(normed='none',le_thresh=le_thresh,ge_thresh=None)
    return coeffs, power


# Main routine - takes SINGLE date
def parallelisation(date):
    contribs=np.ones((len(scales),len(vars)))

    # Calculate spectrum for each variable
    for var in vars:
        print(var,date)
        if var=="sm":
            file_list="soil_moistures"
        elif var=="shfx" and sim=="control":
            file_list="sensible_hfx_control"
        else:
            file_list="surface_vars"
        stash=stash_dict[var]
        try: # the occasional missing file can crash the full portion.
            # Get rid of time component (want 2D array) and RESTRICT SPECTRUM TO CORE SAHEL
            field=load_file(sim+expt,date,hour,file_list,stash).isel(time=0).sel(longitude=slice(348,378),
                                                                                 latitude=slice(9,19))

        except: # if that occurs, fill with NaN. This does mean everything will run with no issue even if other structural issues are present!
            for i in range(len(scales)):
                contribs[i,vars.index(var)]=np.NaN
            continue
        # Apply wavelet transform
        power_con=wavelet_transform(field[-1,:,:])[1] # must be in format time, lat, lon!
        # For each scale in scales, get total spectral power over the Sahel
        for i in range(len(scales)):
            contribs[i,vars.index(var)]=np.sum(power_con[i,:,:])

    df=pd.DataFrame(contribs,columns=["2006%02d%02d_" % (date.month,date.day) + var for var in vars])
    return df

period=pd.date_range("2006-07-25","2006-09-03")
if sim=="control":
    pass
else:
    period=period[:-2] # experiments end on 01/09

if field is None:
    if hour==6: #only field available at 06UTC for sens expts; stash output starts at 07UTC
        vars=["sm"]
    else:
        vars=["sm","lhfx","shfx","t2","q2","lw_nsfc","sw_nsfc","tcc","toaolr","toaswr","lcc"]
    #divvy period up into 4 day blocks - find quicker to do multiple 4 day routines
    period=period[portion*4:4+portion*4]
    
else:
    vars=[field]
    #divvy period up into 16 day blocks - maximum node allocation per job on par-single
    period=period[portion*16:16+portion*16]


p=Pool(len(period))
# Get tables of total power per scale (row index), per field, per day
day_contribs = p.map(parallelisation,period)
# Combine together and save
day_contribs = pd.concat(day_contribs,axis=1)
if len(expt) > 0:
    sim=expt
day_contribs.to_csv("/home/users/bmaybee/LMCS/LMCS_Wafrica_sim/field_scales/Wavelet_power_spectra/Sahel_{}_hr{}-{}.csv".format(sim,hour,portion),index=False)




