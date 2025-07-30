# MCS_SM_scales
Code and data repository supporting Maybee et al, Homogenous soil moisture fields suppress Sahelian MCS frequency (2025). All results based on MetUM model data from simulations run by Helen Burns on Archer2. Data stored on JASMIN collaborative data analysis facility.

Directory contents:
  - sm_paper_plots.ipynb : Notebook used to make all figures in paper and supplementary material, original versions of which are saved in Paper_figs
  - composite_hovmoellers.py : Script for generating composite Hovmoellers either in advance of MCS cores, or after SM patches
  - core_locs_sampling.py : Script for sampling fields about MCS core or SM patch locations
  - get_LMCS_track_data.py : Script for collating raw simpleTrack (see https://github.com/thmstein/simple-track) outputs and updating tables to include rainfall information
  - mcs_core_locs.py : Script for finding convective core locations within 17Z mature MCSs
  - scale_analysis.py : Script for spectral analysis of scale variability
  - SM_filtering_initialisation.py : Script for generating initialisation files for sensitivity experiments
  - sm_scales_utils.py : General shared functions
  - spatial_composites.py : Script for generating 2/3D spatial composites about MCS core or SM patch locations
  - wavelet_power_locs.py: Script for identifying MesoDRY/WET patches using wavelet spectra
  - constants.py, twod.py, wav.py, wclass.py : Package for conducting wavelet transformations, from https://github.com/cornkle/proj_CEH/tree/master/land_wavelet
