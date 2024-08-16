import os, copy
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging

import matplotlib
from IPython.core.pylabtools import figsize, getfigs
import importlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import metpy.constants as mpcon
import metpy.calc as mpcalc
from metpy.units import units

from compute_precip_fields import read_precip
from SensPlotRoutines import background_map

def basin_ens_maps(datea, fhr1, fhr2, config):


    if not os.path.isfile('watershed_precip.nc'):
       logging.warning('  {0} is not present.  Exiting.'.format('watershed_precip.nc'))
       return None

    db = pd.read_csv(filepath_or_buffer=config['metric'].get('basin_huc_file'), \
                     sep = ',', header=None, skipinitialspace=True, quotechar="\"")
    db.columns = ['ID','Name','Size']

    ds = xr.open_dataset('watershed_precip.nc', decode_times=False).rename({'time': 'hour'})
    if ds.attrs['init'] != datea:
       return None

    fff1 = '%0.3i' % fhr1
    fff2 = '%0.3i' % fhr2
    datea_1   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr1)
    date1_str = datea_1.strftime("%Y%m%d%H")
    datea_2   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr2)
    date2_str = datea_2.strftime("%Y%m%d%H")

    ensmat = (ds.precip.sel(hour=slice(fhr2, fhr2)).squeeze()).load()
    ensmat[:,:] = 0.

    prate = (ds.precip.sel(hour=slice(fhr1, fhr2)).squeeze()).load()
    for t in range(1,prate.shape[0]):
       ensmat[:,:] = ensmat[:,:] + np.squeeze(prate[t,:,:])

    e_mean = np.mean(ensmat, axis=0)
    e_std  = np.std(ensmat, axis=0)

    #  Create plots of MSLP and maximum wind for each member, mean and EOF perturbation
    fig = plt.figure(figsize=(13, 7.5))

    colorlist = ("#FFFFFF", "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000", "#FFFF00", \
                 "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000", "#FF00FF", "#9955C9")

    mpcp = np.array([0.0, 0.25, 0.50, 1., 1.5, 2., 4., 6., 8., 12., 16., 24., 32., 64., 96., 97.])

    gdf = gpd.read_file(config['metric'].get('basin_shape_file'))
    gdf = gdf.assign(mean=e_mean)
    gdf = gdf.assign(std=e_std)

    plotBase = config.copy()
    plotBase['subplot']       = 'True'
    plotBase['subrows']       = 1
    plotBase['subcols']       = 2
    plotBase['subnumber']     = 1
    plotBase['grid_interval'] = 180
    plotBase['left_labels'] = 'None'
    plotBase['right_labels'] = 'None'
    plotBase['bottom_labels'] = 'None'
    ax0 = background_map('PlateCarree', -126, -105, 30, 53, plotBase)
    plt.subplots_adjust(wspace=0.02)

    pltm = gdf.plot(ax=ax0, column='mean', cmap=matplotlib.colors.ListedColormap(colorlist), \
                    norm=matplotlib.colors.BoundaryNorm(mpcp,len(mpcp)), edgecolor='black', linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(colorlist), \
                               norm=matplotlib.colors.BoundaryNorm(mpcp,len(mpcp)))
    cbar = fig.colorbar(sm, fraction=0.15, aspect=45., pad=0.02, orientation='horizontal', ticks=mpcp, shrink=0.85, extend='max')
    cbar.set_ticks(mpcp[1:(len(mpcp)-1):2])
    plt.title('Mean')

    plotBase['subnumber']     = 2
    plotBase['left_labels'] = 'None'
    plotBase['right_labels'] = 'None'
    ax1 = background_map('PlateCarree', -126, -105, 30, 53, plotBase)
    plt.subplots_adjust(wspace=0.02)

    #  Plot the standard deviation of the ensemble precipitation
    spcp = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 43.]
    pltm = gdf.plot(ax=ax1, column='std', cmap=matplotlib.colors.ListedColormap(colorlist), \
                    norm=matplotlib.colors.BoundaryNorm(spcp,len(spcp)), edgecolor='black', linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(colorlist), \
                               norm=matplotlib.colors.BoundaryNorm(spcp,len(spcp)))
    cbar = fig.colorbar(sm, fraction=0.15, aspect=45., pad=0.02, orientation='horizontal', ticks=spcp, shrink=0.85, extend='max')
    cbar.set_ticks(spcp[1:(len(spcp)-1):2])
    plt.title('Standard Deviation')

    fig.suptitle('F{0}-F{1} Precipitation ({2}-{3})'.format(fff1, fff2, date1_str, date2_str), fontsize=16)

    outdir = '{0}/std/pbasin'.format(config['locations']['figure_dir'])
    if not os.path.isdir(outdir):
       try:
          os.makedirs(outdir)
       except OSError as e:
          raise e

    plt.savefig('{0}/{1}_f{2}_basin24h_std.png'.format(outdir,datea,fff2),format='png',dpi=120,bbox_inches='tight')
    plt.close(fig)


def precipitation_ens_maps(datea, fhr1, fhr2, config):
    '''
    Function that plots the ensemble precipitation forecast between two forecast hours.

    Attributes:
        datea (string):  initialization date of the forecast (yyyymmddhh format)
        fhr1     (int):  starting forecast hour of the window
        fhr2     (int):  ending forecast hour of the window
        config (dict.):  dictionary that contains configuration options (read from file)
    '''

    dpp = importlib.import_module(config['model']['io_module'])

    lat1 = float(config['fcst_diag'].get('min_lat_precip','30.'))
    lat2 = float(config['fcst_diag'].get('max_lat_precip','52.'))
    lon1 = float(config['fcst_diag'].get('min_lon_precip','-130.'))
    lon2 = float(config['fcst_diag'].get('max_lon_precip','-108.'))

    fff1 = '%0.3i' % fhr1
    fff2 = '%0.3i' % fhr2
    datea_1   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr1)
    date1_str = datea_1.strftime("%Y%m%d%H")
    datea_2   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr2)
    date2_str = datea_2.strftime("%Y%m%d%H")
    fint      = int(config['model'].get('fcst_hour_int','12'))
    g1        = dpp.ReadGribFiles(datea, fhr1, config)

    vDict = {'latitude': (lat1-0.00001, lat2+0.00001), 'longitude': (lon1-0.00001, lon2+0.00001),
             'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
    vDict = g1.set_var_bounds('precipitation', vDict)

    ensmat = read_precip(datea, fhr1, fhr2, config, vDict)

    #  Scale all of the rainfall to mm and to a 24 h precipitation
    ensmat[:,:,:] = ensmat[:,:,:] * 24. / float(fhr2-fhr1)

    e_mean = np.mean(ensmat, axis=0)
    e_std  = np.std(ensmat, axis=0)

    #  Create basic figure, including political boundaries and grid lines
    fig = plt.figure(figsize=(11,6.5), constrained_layout=True)

    colorlist = ("#FFFFFF", "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000", "#FFFF00", \
                 "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000", "#FF00FF", "#9955C9")

    plotBase = config.copy()
    plotBase['subplot']       = 'True'
    plotBase['subrows']       = 1
    plotBase['subcols']       = 2
    plotBase['subnumber']     = 1
    plotBase['grid_interval'] = config['fcst_diag'].get('grid_interval', 5)
    plotBase['left_labels'] = 'True'
    plotBase['right_labels'] = 'None'
    ax1 = background_map(config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

    #  Plot the mean precipitation map
    mpcp = [0.0, 0.25, 0.50, 1., 1.5, 2., 4., 6., 8., 12., 16., 24., 32., 64., 96., 97.]
    norm = matplotlib.colors.BoundaryNorm(mpcp,len(mpcp))
    pltf1 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_mean,mpcp,norm=norm,extend='max', \
                         cmap=matplotlib.colors.ListedColormap(colorlist), transform=ccrs.PlateCarree())

    cbar = plt.colorbar(pltf1, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mpcp)
    cbar.set_ticks(mpcp[1:(len(mpcp)-1):2])

    plt.title('Mean')

    plotBase['subnumber']     = 2
    plotBase['left_labels'] = 'None'
    plotBase['right_labels'] = 'None'
    ax2 = background_map(config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

    #  Plot the standard deviation of the ensemble precipitation
    spcp = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 43.]
    norm = matplotlib.colors.BoundaryNorm(spcp,len(spcp))
    pltf2 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_std,spcp,norm=norm,extend='max', \
                         cmap=matplotlib.colors.ListedColormap(colorlist), transform=ccrs.PlateCarree())

    cbar = plt.colorbar(pltf2, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=spcp)
    cbar.set_ticks(spcp[1:(len(spcp)-1)])

    plt.title('Standard Deviation')

    fig.suptitle('F{0}-F{1} Precipitation ({2}-{3})'.format(fff1, fff2, date1_str, date2_str), fontsize=16)

    outdir = '{0}/std/pcp'.format(config['locations']['figure_dir'])
    if not os.path.isdir(outdir):
       try:
          os.makedirs(outdir)
       except OSError as e:
          raise e

    plt.savefig('{0}/{1}_f{2}_pcp24h_std.png'.format(outdir,datea,fff2),format='png',dpi=120,bbox_inches='tight')
    plt.close(fig)


def read_ivt(datea, fhr, config, vDict):
   '''
   Routine that reads either the IVT u/v/magnitude if those variables are present in a file;
   otherwise, compute these quantities from basic meteorological variables.  The subroutine 
   returns the ensemble IVT u/v/magnitude fields. 

   Attributes:
       datea (string):  Initialization date (yyyymmddhh format)
       fhr (integer):   forecast hour
       config (dict.):  dictionary that contains configuration options (read from file)
       vDict (dict.):   dictionary that contains domain-specific options
   '''

   dpp = importlib.import_module(config['model']['io_module'])
   gf = dpp.ReadGribFiles(datea, fhr, config)
   ivtu = gf.create_ens_array('temperature', gf.nens, vDict)
   ivtv = gf.create_ens_array('temperature', gf.nens, vDict)
   ivtt = gf.create_ens_array('temperature', gf.nens, vDict)

   if 'ivt' in gf.var_dict:

      #  Read each ensemble member from file
      for n in range(gf.nens):
         ivtu[n,:,:] = gf.read_grib_field('ivtu', n, vDict)
         ivtv[n,:,:] = gf.read_grib_field('ivtv', n, vDict)
         ivtt[n,:,:] = gf.read_grib_field('ivt',  n, vDict)

   else:

      tDict = copy.deepcopy(vDict)
      tDict['isobaricInhPa'] = (300, 1000)
      tDict = gf.set_var_bounds('temperature', tDict)
      wDict = copy.deepcopy(vDict)
      wDict['isobaricInhPa'] = (300, 1000)
      wDict = gf.set_var_bounds('zonal_wind', wDict)

      for n in range(gf.nens):

         #  Read wind/temperature data 
         uwnd = gf.read_grib_field('zonal_wind', n, wDict) * units('m / sec')
         vwnd = gf.read_grib_field('meridional_wind', n, wDict) * units('m / sec')

         tmpk = np.squeeze(gf.read_grib_field('temperature', n, tDict)) * units('K')
         pres = (tmpk.isobaricInhPa.values * units.hPa).to(units.Pa)

         #  Read or calculate specific humidity based on what is in the original
         if gf.has_specific_humidity:
            qvap = np.squeeze(gf.read_grib_field('specific_humidity', n, tDict)) * units('dimensionless')
         else:
            relh = np.minimum(np.maximum(gf.read_grib_field('relative_humidity', n, tDict), 0.01), 100.0) * units('percent')
            qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)

         #  Integrate water vapor over the pressure levels
         ivtu[n,:,:] = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
         ivtv[n,:,:] = np.abs(np.trapz(vwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
         ivtt[n,:,:] = np.sqrt(ivtu[n,:,:]**2 + ivtv[n,:,:]**2)

   return(ivtu, ivtv, ivtt)
