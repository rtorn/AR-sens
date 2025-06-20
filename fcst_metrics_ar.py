import os, sys, glob
import pandas as pd
import numpy as np
import xarray as xr
import json
import numpy as np
import datetime as dt
import logging
import configparser
import geopandas as gpd

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

from eofs.standard import Eof
from eofs.xarray import Eof as Eof_xarray

from SensPlotRoutines import background_map
from fcst_diag import read_ivt
import grid_calc

def great_circle(lon1, lat1, lon2, lat2):
    '''
    Function that computes the distance between two lat/lon pairs.  The result of this function 
    is the distance in kilometers.

    Attributes
        lon1 (float): longitude of first point
        lat1 (float): latitude of first point
        lon2 (float): longitude of second point.  Can be an array
        lat2 (float): latitude of second point.  Can be an array
    '''

    dist = np.empty(lon2.shape)

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2[:] = np.radians(lon2[:])
    lat2[:] = np.radians(lat2[:])

    dist[:] = np.sin(lat1) * np.sin(lat2[:]) + np.cos(lat1) * np.cos(lat2[:]) * np.cos(lon1 - lon2[:])

    return 6371. * np.arccos(np.minimum(dist,1.0))

class ComputeForecastMetrics:
    '''
    Function that computes ensemble-based estimates of TC forecast metrics based on the information
    within the configuration file.  Each of these metrics is stored in a seperate netCDF file that
    is used to compute the sensitivity.

    Attributes:
        datea (string): initialization date of the forecast (yyyymmddhh format)
        config (dict.):  dictionary that contains configuration options (read from file)
    '''

    def __init__(self, datea, config):

        #  Define class-specific variables
        self.fhr = None
        self.deg2rad = 0.01745
        self.earth_radius = 6378388.
        self.missing = -9999.
        self.deg2km = self.earth_radius * np.radians(1)

        self.datea_str = datea
        self.datea = dt.datetime.strptime(datea, '%Y%m%d%H')
        self.datea_s = self.datea.strftime("%m%d%H%M")
        self.outdir = config['locations']['output_dir']

        self.config = config

        self.metlist = []
        self.dpp = importlib.import_module(config['model']['io_module'])

        if self.config['metric'].get('precipitation_mean_metric', 'True') == 'True':
           self.__precipitation_mean()

        #  Compute precipitation EOF metric
        if self.config['metric'].get('precipitation_eof_metric', 'True') == 'True':
           self.__precipitation_eof()

        #  Compute IVT EOF metric
        if self.config['metric'].get('ivt_eof_metric', 'False') == 'True':
           self.__ivt_eof()

        #  Compute IVT Landfall EOF metric
        if self.config['metric'].get('ivt_landfall_metric', 'False') == 'True':
           self.__ivt_landfall_eof()

        #  Compute basin EOF metric
        if self.config['metric'].get('basin_metric', 'False') == 'True':
           self.__precip_basin_eof()

        #  Compute SLP EOF metric
        if self.config['metric'].get('slp_eof_metric', 'False') == 'True':
           self.__slp_eof()

        #  Compute Height EOF metric
        if self.config['metric'].get('hght_eof_metric', 'False') == 'True':
           self.__hght_eof()

        #  Compute PV EOF metric
        if self.config['metric'].get('pvort_eof_metric', 'False') == 'True':
           self.__pvort_eof()

        #  Compute temperature EOF metric
        if self.config['metric'].get('temp_eof_metric', 'False') == 'True':
           self.__temp_eof()


    def get_metlist(self):
        '''
        Function that returns a list of metrics being considered
        '''
        return self.metlist


    def __background_map(self, ax, lat1, lon1, lat2, lon2):
        '''
        Function that creates a background map for plotting.

        Attributes:
           ax    (axis):  figure axes being used for the plot
           lat1 (float):  minimum latitude of the plot
           lon1 (float):  minimum longitude of the plot
           lat2 (float):  maximum latitude of the plot
           lon2 (float):  maximum longitude of the plot
        '''


        gridInt = 5

        states = NaturalEarthFeature(category="cultural", scale="50m",
                                     facecolor="none",
                                     name="admin_1_states_provinces_shp")
        ax.add_feature(states, linewidth=0.5, edgecolor="black")
        ax.coastlines('50m', linewidth=1.0)
        ax.add_feature(cartopy.feature.LAKES, facecolor='None', linewidth=1.0, edgecolor='black')
        ax.add_feature(cartopy.feature.BORDERS, facecolor='None', linewidth=1.0, edgecolor='black')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=1, color='gray', alpha=0.5, linestyle='-')
        gl.top_labels = None
        gl.left_labels = None
        gl.xlocator = mticker.FixedLocator(np.arange(10.*np.floor(0.1*lon1),10.*np.ceil(0.1*lon2)+1.,gridInt))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size': 12, 'color': 'gray'}
        gl.ylocator = mticker.FixedLocator(np.arange(10.*np.floor(0.1*lat1),10.*np.ceil(0.1*lat2)+1.,gridInt))
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'size': 12, 'color': 'gray'}

        ax.set_extent([lon1, lon2, lat1, lat2], ccrs.PlateCarree())

        return ax

    def __ivt_eof(self):
        '''
        Function that computes the IVT EOF metric, which is calculated by taking the EOF of 
        the ensemble IVT forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean IVT pattern 
        along with the IVT perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('ivt_metric_loc'),self.datea_str)):

           print(infull)

           #  Read the text file that contains information on the precipitation metric
           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr = int(conf['definition'].get('forecast_hour'))
              metname = conf['definition'].get('metric_name','ivteof')
              eofn = int(conf['definition'].get('eof_number',1))
              lat1 = float(conf['definition'].get('latitude_min'))
              lat2 = float(conf['definition'].get('latitude_max'))
              lon1 = float(conf['definition'].get('longitude_min'))
              lon2 = float(conf['definition'].get('longitude_max'))
              vecmet = eval(conf['definition'].get('vector',self.config['metric'].get('ivt_land_vector','False')))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT EOF'.format(infull))
              return None

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ivtu, ivtv, ivtm = read_ivt(self.datea_str, fhr, self.config, vDict)
           ivtm_mean = np.mean(ivtm, axis=0)

           if vecmet:

              ivt = xr.DataArray(name='ensemble_data', data=np.zeros([g1.nens, 2, len(ivtm_mean[:,0]), len(ivtm_mean[0,:])]), \
                                 dims=['ensemble', 'component', 'latitude', 'longitude'], \
                                 coords={'ensemble': [i for i in range(g1.nens)], 'latitude': ivtm_mean.latitude.values, 'longitude': ivtm_mean.longitude.values})

              ivt[:,0,:,:] = ivtu[:,:,:]
              ivt[:,1,:,:] = ivtv[:,:,:]           
   
              ivt_mean = np.mean(ivt, axis=0)
              for n in range(g1.nens):
                 ivt[n,:,:,:] = ivt[n,:,:,:] - ivt_mean[:,:,:]

              #  Compute the EOF of the precipitation pattern and then the PCs
              if self.config['model'].get('grid_type','LatLon') == 'LatLon':

                 coslat = np.cos(np.deg2rad(ivtm.latitude.values)).clip(0., 1.)
                 wgts = np.sqrt(coslat)[..., np.newaxis]
                 solver = Eof_xarray(ivt.rename({'ensemble': 'time'}), weights=wgts)

              else:

                 solver = Eof_xarray(ivt.rename({'ensemble': 'time'}))

           else:

              for n in range(g1.nens):
                 ivtm[n,:,:] = ivtm[n,:,:] - ivtm_mean[:,:]

              #  Compute the EOF of the precipitation pattern and then the PCs
              if self.config['model'].get('grid_type','LatLon') == 'LatLon':

                 coslat = np.cos(np.deg2rad(ivtm.latitude.values)).clip(0., 1.)
                 wgts = np.sqrt(coslat)[..., np.newaxis]
                 solver = Eof_xarray(ivtm.rename({'ensemble': 'time'}), weights=wgts)

              else:

                 solver = Eof_xarray(ivtm.rename({'ensemble': 'time'}))

           pcout  = solver.pcs(npcs=eofn, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the IVT pattern associated with a 1 PC perturbation
           if vecmet:

              divu = np.zeros(ivtm_mean.shape)
              divv = np.zeros(ivtm_mean.shape)

              for n in range(g1.nens):
                 divu[:,:] = divu[:,:] + ivt[n,0,:,:] * pc1[n]
                 divv[:,:] = divv[:,:] + ivt[n,1,:,:] * pc1[n]

              divu[:,:] = divu[:,:] / float(g1.nens)
              divv[:,:] = divv[:,:] / float(g1.nens)

              if np.sum(divu+divv) < 0.0:
                 divu[:,:] = -divu[:,:]
                 divv[:,:] = -divv[:,:]
                 pc1[:]    = -pc1[:]
           
           else:

              divm = np.zeros(ivtm_mean.shape)

              for n in range(g1.nens):
                 divm[:,:] = divm[:,:] + ivtm[n,:,:] * pc1[n]

              divm[:,:] = divm[:,:] / float(g1.nens)

              if np.sum(divm) < 0.0:
                 divm[:,:] = -divm[:,:]
                 pc1[:]    = -pc1[:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#FFFFFF", "#FFFC00", "#FFE100", "#FFC600", "#FFAA00", "#FF7D00", \
                        "#FF4B00", "#FF1900", "#E60015", "#B3003E", "#80007B", "#570088")

           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           mivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
           norm = matplotlib.colors.BoundaryNorm(mivt,len(mivt))
           pltf = plt.contourf(ivtm.longitude.values,ivtm.latitude.values,ivtm_mean,mivt,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           if vecmet:

              divm[:,:] = np.sqrt(divu**2+divv**2)
              divu[:,:] = np.where(divm >= 30.0, divu, np.nan)
              divv[:,:] = np.where(divm >= 30.0, divv, np.nan)
              sout = (2500./30.) * (np.max(ivtm.latitude.values)-np.min(ivtm.latitude.values))
              qo1 = ax.quiver(ivtm.longitude.values,ivtm.latitude.values,divu,divv, \
                         scale_units='height', scale=sout, width=0.005, pivot='mid', color='black', minlength=0)
              l, b, w, h = ax.get_position().bounds
              qk = ax.quiverkey(qo1, l+w-0.05, b+0.07, 100, '100', labelpos='E', coordinates='figure')

           else:

              ivtfac = np.floor(np.log10(np.max(np.abs(divm))))
              cntrs = np.array([-9., -8., -7., -6., -5., -4., -3., -2., -1.5, -1., -0.8, -0.6, 0.6, 0.8, 1., 1.5, 2., 3., 4., 5., 6., 7., 8., 9.]) * (10**ivtfac)
              pltm = plt.contour(ivtm.longitude.values,ivtm.latitude.values,divm,cntrs,linewidths=1.5, \
                                   transform=ccrs.PlateCarree(),colors='k',zorder=10)
              cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mivt)
           cbar.set_ticks(mivt[1:(len(mivt)-1)])

           fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           plt.title("{0} {1} hour IVT, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'IVT PC', 'FORECAST_METRIC_SHORT_NAME': 'ivteof', 'FORECAST_HOUR': int(fhr), \
                      'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, 'LONGITUDE2': lon2, 'VECTOR': str(vecmet), 'EOF_NUMBER': int(eofn)}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ivtm.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ivtm.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'kg m**-1 s**-1', 'description': 'IVT ensemble mean'}, 'data': ivtm_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'kg m**-1 s**-1', 'description': 'IVT EOF pattern'}, 'data': divm}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'IVT PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr, metname))


    def __ivt_landfall_eof(self):
        '''
        Function that computes the IVT EOF metric, which is calculated by taking the EOF of 
        the ensemble IVT forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean IVT pattern 
        along with the IVT perturbation that is consistent with the first EOF. 
        '''

        colorlist = ("#FFFFFF", "#FFFC00", "#FFE100", "#FFC600", "#FFAA00", "#FF7D00", \
                     "#FF4B00", "#FF1900", "#E60015", "#B3003E", "#80007B", "#570088")

        try:
           dcoa = pd.read_csv(self.config['metric'].get('coast_points_file'), sep = '\\s+', header=None, names=['latitude', 'longitude'])
           latcoa = dcoa['latitude'].values
           loncoa = dcoa['longitude'].values
           f = open(self.config['metric'].get('coast_points_file'), 'r')
        except IOError:
           logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(self.config['metric'].get('coast_points_file')))
           return None

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('ivt_land_metric_loc'),self.datea_str)):

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr1 = int(conf['definition'].get('forecast_hour1',self.config['metric'].get('ivt_land_forecast_hour1',0)))
              fhr2 = int(conf['definition'].get('forecast_hour2',self.config['metric'].get('ivt_land_forecast_hour2',120)))
              metname = conf['definition'].get('metric_name','ivtland')
              eofn = int(conf['definition'].get('eof_number',1))
              latcoa1 = float(conf['definition'].get('latitude_min',self.config['metric'].get('ivt_land_latitude_min',25.)))
              latcoa2 = float(conf['definition'].get('latitude_max',self.config['metric'].get('ivt_land_latitude_max',55.)))
              vecmet = eval(conf['definition'].get('vector',self.config['metric'].get('ivt_land_vector','False')))
              adapt = eval(conf['definition'].get('adapt',self.config['metric'].get('ivt_land_adapt','False')))
              ivtmin = float(conf['definition'].get('adapt_ivt_min',self.config['metric'].get('ivt_land_adapt_min',200.)))
              latbuff = float(conf['definition'].get('adapt_ivt_lat_buff',self.config['metric'].get('ivt_land_adapt_lat_buff',0.)))
              timebuff = int(conf['definition'].get('adapt_ivt_hour_buff',self.config['metric'].get('ivt_land_adapt_hour_buff',0)))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(infull))
              continue

           latlist = []
           lonlist = []
           for i in range(len(latcoa)):
              if latcoa[i] >= latcoa1 and latcoa[i] <= latcoa2:
                 latlist.append(latcoa[i])
                 lonlist.append(loncoa[i])

           if eval(self.config['model'].get('flip_lon','False')):
              for i in range(len(loncoa)):
                 loncoa[i] = (loncoa[i] + 360.) % 360.
              for i in range(len(lonlist)):
                 lonlist[i] = (lonlist[i] + 360.) % 360.

           lat1 = np.min(latlist)
           lat2 = np.max(latlist)
           lon1 = np.min(lonlist)
           lon2 = np.max(lonlist)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}

           if eval(self.config['metric'].get('metric_from_fields','False')) and \
              os.path.exists('{0}/{1}_f{2}_ivt_ens.nc'.format(self.config['locations']['work_dir'],self.datea_str,'%0.3i' % fhr1)):

              vDict['from_field'] = 'True'
              g1 = []

           else:

              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1, self.config)
              vDict = g1.set_var_bounds('temperature', vDict)

           ivtarr = self.__ivt_landfall(fhr1, fhr2, latlist, lonlist, vDict, g1)
           e_mean = np.mean(ivtarr, axis=0)
           nens   = ivtarr.shape[0]

           if adapt:

              ntime = len(e_mean.fcst_hour.values)
              nlat  = len(e_mean.latitude.values)

              estd   = np.mean(ivtarr[:,2,:,:], axis=0)
              stdmax = estd.max()
              maxloc = np.where(estd == stdmax)
              icen   = int(maxloc[1])
              jcen   = int(maxloc[0])
              timec  = e_mean.fcst_hour.values[icen]
              latc   = e_mean.latitude.values[jcen]

              if np.amax(e_mean[2,:,:]) < ivtmin and e_mean[2,jcen,icen] < ivtmin:
                 logging.error('  IVT landfall metric center point is below minimum.  Skipping metric.')
                 continue

              fmgrid = e_mean[0,:,:].squeeze().copy()
              fmgrid[:,:] = 0.0
              fmgrid[jcen,icen] = 1.0

              iloc       = np.zeros(ntime*nlat, dtype=int)
              jloc       = np.zeros(ntime*nlat, dtype=int)
              nloc       = 0
              iloc[nloc] = icen
              jloc[nloc] = jcen

              fhr1       = timec
              fhr2       = timec
              lat1       = latc
              lat2       = latc

              k = 0
              while k <= nloc:

                 fhr1 = min(fhr1,e_mean.fcst_hour.values[iloc[k]])
                 fhr2 = max(fhr2,e_mean.fcst_hour.values[iloc[k]])
                 lat1 = min(lat1,e_mean.latitude.values[jloc[k]])
                 lat2 = max(lat2,e_mean.latitude.values[jloc[k]])

                 for i in range(max(iloc[k]-1,0), min(iloc[k]+2,ntime)):
                    for j in range(max(jloc[k]-1,0), min(jloc[k]+2,nlat)):
                       if e_mean[2,j,i] >= ivtmin and fmgrid[j,i] < 1.0:
                          nloc = nloc + 1
                          iloc[nloc] = i
                          jloc[nloc] = j
                          fmgrid[j,i] = 1.0

                 k = k + 1

              #  Evaluate whether the forecast metric grid has enough land points
              if k == 0 or fhr1 == fhr2 or lat1 == lat2:
                 logging.error('  IVT landfall metric does not have any points above minimum.  Skipping metric.')
                 continue

              fhr1 = max(fhr1-timebuff, min(e_mean.fcst_hour.values))
              fhr2 = min(fhr2+timebuff, max(e_mean.fcst_hour.values))
              lat1 = max(lat1-latbuff,  min(e_mean.latitude.values))
              lat2 = min(lat2+latbuff,  max(e_mean.latitude.values))
              ivtarr = ivtarr.sel(latitude=slice(lat2,lat1), fcst_hour=slice(fhr1,fhr2))
              e_mean = e_mean.sel(latitude=slice(lat2,lat1), fcst_hour=slice(fhr1,fhr2))

              latlist = []
              lonlist = []
              for i in range(len(latcoa)):
                 if latcoa[i] >= lat1 and latcoa[i] <= lat2:
                    latlist.append(latcoa[i])
                    lonlist.append(loncoa[i])

           for n in range(nens):
              ivtarr[n,:,:,:] = ivtarr[n,:,:,:] - e_mean[:,:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if vecmet:
              solver = Eof_xarray(ivtarr[:,0:2,:,:].squeeze().rename({'ensemble': 'time'}))
           else:
              solver = Eof_xarray(ivtarr[:,2,:,:].squeeze().rename({'ensemble': 'time'}))
           pcout  = solver.pcs(npcs=eofn, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the IVT pattern associated with a 1 PC perturbation
           divt = np.zeros(np.squeeze(e_mean[2,:,:]).shape)
           for n in range(nens):
              divt[:,:] = divt[:,:] + ivtarr[n,2,:,:] * pc1[n]

           divt[:,:] = divt[:,:] / float(nens)

           if vecmet:

              divu = np.zeros(np.squeeze(e_mean[0,:,:]).shape)
              divv = np.zeros(np.squeeze(e_mean[1,:,:]).shape)
              for n in range(nens):
                 divu[:,:] = divu[:,:] + ivtarr[n,0,:,:] * pc1[n]
                 divv[:,:] = divv[:,:] + ivtarr[n,1,:,:] * pc1[n]

              divu[:,:] = divu[:,:] / float(nens)
              divv[:,:] = divv[:,:] / float(nens)

              if np.sum(divu) + np.sum(divv) < 0.0:
                 divu[:,:] = -divu[:,:]
                 divv[:,:] = -divv[:,:]
                 pc1[:]    = -pc1[:]

           else:

              if np.sum(divt) < 0.0:
                 divt[:,:] = -divt[:,:]
                 pc1[:]    = -pc1[:]

           fig = plt.figure(figsize=(10, 6))

           ax0 = fig.add_subplot(121)
           ax0.set_facecolor("gainsboro")

           mivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
           norm = matplotlib.colors.BoundaryNorm(mivt,len(mivt))
           pltf = ax0.contourf(ivtarr.fcst_hour.values,ivtarr.latitude.values,np.squeeze(e_mean[2,:,:]),mivt, \
                                cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           if vecmet:

              divu[:,:] = np.where(np.abs(divt) >= 30.0, divu, np.nan)
              divv[:,:] = np.where(np.abs(divt) >= 30.0, divv, np.nan)
              qo1 = ax0.quiver(ivtarr.fcst_hour.values,ivtarr.latitude.values,divu,divv, \
                         scale_units='height', scale=2500, width=0.005, pivot='mid', color='black', minlength=0)
              qk = ax0.quiverkey(qo1, 0.45, 0.9, 100, '100', labelpos='E', coordinates='figure')

           else:

              ivtfac = np.max(abs(divt))
              if ivtfac < 60:
                 cntrs = np.array([-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
              elif ivtfac >= 60 and ivtfac < 300:
                 cntrs = np.array([-270, -240, -210, -180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180, 210, 240, 270])
              else:
                 cntrs = np.array([-500, -400, -300, -200, -100, 100, 200, 300, 400, 500])

              pltm = ax0.contour(ivtarr.fcst_hour.values,ivtarr.latitude.values,divt,cntrs,linewidths=1.5, colors='k', zorder=10)
              cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           ax0.set_xlim([np.min(ivtarr.fcst_hour.values), np.max(ivtarr.fcst_hour.values)])
           ax0.set_ylim([np.min(latcoa), np.max(latcoa)])

           plt.xticks(np.arange(fhr1, fhr2, step=12.))
           plt.xlabel('Forecast Hour')
           plt.ylabel('Latitude')

           plotBase = self.config.copy()
           plotBase['subplot']       = 'True'
           plotBase['subrows']       = 1
           plotBase['subcols']       = 2
           plotBase['subnumber']     = 2
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'None'
           plotBase['right_labels'] = 'None'
           plotBase['bottom_labels'] = 'None'
           ax1 = background_map('PlateCarree', -133, -111, np.min(latcoa), np.max(latcoa), plotBase)

           ax1.plot(loncoa, latcoa, 'o', color='black', markersize=6, transform=ccrs.PlateCarree())

           if eofn == 1:
              fracvar = solver.varianceFraction(neigs=1).data
           else:
              fracvar = solver.varianceFraction(neigs=eofn)[-1].data
           plt.suptitle("{0} {1}-{2} hour IVT, {3} of variance".format(str(self.datea_str),fhr1,fhr2,'%4.3f' % fracvar))

           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.13, orientation='horizontal', cax=fig.add_axes([0.15, 0.01, 0.7, 0.025]))
           cbar.set_ticks(mivt[1:(len(mivt)-1)])

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],'%0.3i' % fhr2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'IVT Landfall PC', 'FORECAST_METRIC_SHORT_NAME': 'ivtleof', \
                      'FORECAST_HOUR1': int(fhr1), 'FORECAST_HOUR2': int(fhr2), 'LATITUDE1': lat1, 'LATITUDE2': lat2, \
                      'ADAPT': str(adapt), 'ADAPT_IVT_MIN': ivtmin, 'VECTOR': str(vecmet), 'EOF_NUMBER': int(eofn), 'VAR_FRACTION': fracvar}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': nens}, 'data_vars': {}}
           f_met['coords']['forecast_hour'] = {'dims': ('forecast_hour'), 'attrs': {'units': 'hr', 'description': 'forecast hour'}, 'data': ivtarr.fcst_hour.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of landfall points'}, 'data': ivtarr.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}
           f_met['data_vars']['longitude'] = {'dims': ('locations',), 'attrs': {'units': 'degrees', 'description': 'longitude of landfall points'}, 'data': lonlist}
           endict['longitude'] = {'dtype': 'float32'}

           if vecmet:
              print('implement')
           else:
              f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'forecast_hour'), 'attrs': {'units': 'kg m**-1 s**-1', 'description': 'IVT Landfall ensemble mean'}, 'data': np.squeeze(e_mean[2,:,:].data)}
              endict['ensemble_mean'] = {'dtype': 'float32'}
              f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'forecast_hour'), 'attrs': {'units': 'kg m**-1 s**-1', 'description': 'IVT Landfall EOF pattern'}, 'data': divt}
              endict['EOF_pattern'] = {'dtype': 'float32'}       
 
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'IVT Landfall PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr2,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr2, metname))


    def __read_ivt(self, fhr, vDict):

        gf = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)
        ivtout = gf.create_ens_array('temperature', gf.nens, vDict)

        if 'ivt' in gf.var_dict:

           for n in range(gf.nens):
              ivtout[n,:,:] = gf.read_grib_field('ivt', n, vDict)

        else:

           fDict = vDict.copy()
           fDict['isobaricInhPa'] = (300, 1000)
           fDict = gf.set_var_bounds('temperature', fDict)

           for n in range(gf.nens):

              uwnd = gf.read_grib_field('zonal_wind', n, fDict) * units('m / sec')
              vwnd = gf.read_grib_field('meridional_wind', n, fDict) * units('m / sec')

              tmpk = np.squeeze(gf.read_grib_field('temperature', n, fDict)) * units('K')
              pres = (tmpk.isobaricInhPa.values * units.hPa).to(units.Pa)

              if gf.has_specific_humidity:
                 qvap = np.squeeze(gf.read_grib_field('specific_humidity', n, fDict)) * units('dimensionless')
              else:
                 relh = np.minimum(np.maximum(gf.read_grib_field('relative_humidity', n, fDict), 0.01), 100.0) * units('percent')
                 qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)

              #  Integrate water vapor over the pressure levels
              usum = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
              vsum = np.abs(np.trapz(vwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
              ivtout[n,:,:] = np.sqrt(usum[:,:]**2 + vsum[:,:]**2)

        return(ivtout)


    def __ivt_landfall(self, fhr1, fhr2, latlist, lonlist, vDict, gf):

        if eval(vDict.get('from_field','False')):

           ds = xr.open_dataset('{0}/{1}_f{2}_ivt_ens.nc'.format(self.config['locations']['work_dir'],self.datea_str,'%0.3i' % fhr1))
           latitude  = ds.ensemble_data.latitude.values
           longitude = ds.ensemble_data.longitude.values
           nens      = ds.ensemble_data.shape[0]

        else:

           ensmat    = gf.create_ens_array('temperature', gf.nens, vDict)
           latitude  = ensmat.latitude.values
           longitude = ensmat.longitude.values
           nens      = gf.nens

        fhrvec = np.arange(fhr1, fhr2+int(self.config['model']['fcst_hour_int']), int(self.config['model']['fcst_hour_int']))

        ivtarr = xr.DataArray(name='ensemble_data', data=np.zeros([nens, 3, len(latlist), len(fhrvec)]), dims=['ensemble', 'component', 'latitude', 'fcst_hour'], \
                              coords={'ensemble': [i for i in range(nens)], 'fcst_hour': fhrvec, 'latitude': latlist})

        vecloc = len(np.shape(latitude)) == 1

        if not vecloc:

           xloc = np.zeros(len(latlist), dtype=int)
           yloc = np.zeros(len(latlist), dtype=int)

           for i in range(len(latlist)):

              abslat = np.abs(latitude-latlist[i])
              abslon = np.abs(longitude-lonlist[i])
              c = np.maximum(abslon, abslat)

              ([yloc[i]], [xloc[i]]) = np.where(c == np.min(c))

        for t in range(len(fhrvec)):

           ivtu, ivtv, ivtm = read_ivt(self.datea_str, int(fhrvec[t]), self.config, vDict)

           if vecloc:

              for i in range(len(latlist)):
                 ivtarr[:,0,i,t] = ivtu.sel(latitude=slice(latlist[i], latlist[i]), \
                                            longitude=slice(lonlist[i], lonlist[i])).squeeze()
                 ivtarr[:,1,i,t] = ivtv.sel(latitude=slice(latlist[i], latlist[i]), \
                                            longitude=slice(lonlist[i], lonlist[i])).squeeze() 
                 ivtarr[:,2,i,t] = ivtm.sel(latitude=slice(latlist[i], latlist[i]), \
                                            longitude=slice(lonlist[i], lonlist[i])).squeeze()

           else:

              for i in range(len(latlist)):
                 ivtarr[:,0,i,t] = ivtu.sel(lat=yloc[i], lon=xloc[i]).squeeze().data
                 ivtarr[:,1,i,t] = ivtv.sel(lat=yloc[i], lon=xloc[i]).squeeze().data
                 ivtarr[:,2,i,t] = ivtm.sel(lat=yloc[i], lon=xloc[i]).squeeze().data

        del latitude,longitude

        return(ivtarr)


    def __precipitation_mean(self):
        '''
        Function that computes the average precipitation over a polygon using the 
        domain defined by the user in a text file.  
        The function also plots a figure showing the ensemble mean and standard deviation of
        precipitation pattern along with the metric area.
        '''

        search_max = 250.

        logging.warning('  Mean Precipitation Metrics:')

        for infile in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('precip_mean_loc'),self.datea_str)):

           fint = int(self.config['metric'].get('fcst_int',self.config['model']['fcst_hour_int']))

           try:
              conf = configparser.ConfigParser()
              conf.read(infile)
              fhr1 = int(conf['definition'].get('forecast_hour1',self.config['metric'].get('precip_mean_forecast_hour1','48')))
              fhr2 = int(conf['definition'].get('forecast_hour2',self.config['metric'].get('precip_mean_forecast_hour2','120')))
              lat1 = float(conf['definition'].get('latitude_min',30))
              lat2 = float(conf['definition'].get('latitude_max',52))
              lon1 = float(conf['definition'].get('longitude_min',-130))
              lon2 = float(conf['definition'].get('longitude_max',-108))
              latc = float(conf['definition'].get('latitude_center'))
              lonc = float(conf['definition'].get('longitude_center'))
              maxf = float(conf['definition'].get('max_std_dev_factor',0.667))
              adapt = eval(conf['definition'].get('adapt',self.config['metric'].get('precip_mean_adapt','False')))
              time_adapt = eval(conf['definition'].get('time_adapt',self.config['metric'].get('precip_mean_time_adapt','False')))
              time_dbuff = float(conf['definition'].get('time_adapt_domain',self.config['metric'].get('precip_mean_time_adapt_domain',2.0)))
              time_freq = int(conf['definition'].get('time_adapt_freq',self.config['metric'].get('precip_mean_time_adapt_freq',6)))
              pcpmin = float(conf['definition'].get('adapt_pcp_min',self.config['metric'].get('precip_mean_adapt_pcp_min','12.7')))
              lmaskmin = float(conf['definition'].get('land_mask_minimum',self.config['metric'].get('land_mask_minimum','0.2')))
              mask_land = eval(conf['definition'].get('land_mask',self.config['metric'].get('precip_mean_land_mask','False')))
              frozen = eval(conf['definition'].get('frozen_mask',self.config['metric'].get('precip_mean_frozen_mask','False')))
              metname = conf['definition'].get('metric_name','pcp')
           except:
              logging.warning('  {0} does not exist.  Using parameter and/or default values'.format(infile))

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           datea_1   = self.datea + dt.timedelta(hours=fhr1)
           date1_str = datea_1.strftime("%Y%m%d%H")
           datea_2   = self.datea + dt.timedelta(hours=fhr2)
           date2_str = datea_2.strftime("%Y%m%d%H")

           #  Read the total precipitation, scale to a 24 h value 
           vDict = {'latitude': (lat1-0.00001, lat2+0.00001), 'longitude': (lon1-0.00001, lon2+0.00001),
                       'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
           ensmat = self.__read_precip(fhr1, fhr2, self.config, vDict)
           ensmat[:,:,:] = ensmat[:,:,:] * 24. / float(fhr2-fhr1)

           lonarr, latarr = np.meshgrid(ensmat.longitude.values, ensmat.latitude.values)
           cdist = great_circle(lonc, latc, lonarr, latarr)
           nlon  = len(ensmat.longitude.values)
           nlat  = len(ensmat.latitude.values)

           e_mean = np.mean(ensmat, axis=0)
           e_std  = np.std(ensmat, axis=0)

           #  Blank out all locations outside of search radius
           cdist = np.where(cdist <= search_max, 1.0, 0.0)
           estd_mask = e_std[:,:] * cdist[:,:]

           stdmax = estd_mask.max()
           maxloc = np.where(estd_mask == stdmax)
           icen   = int(maxloc[1])
           jcen   = int(maxloc[0])

           stdmax = stdmax * maxf

           fmgrid = np.zeros(e_mean.shape)
           fmgrid[jcen,icen] = 1.0

           iloc       = np.zeros(nlon*nlat, dtype=int)
           jloc       = np.zeros(nlon*nlat, dtype=int)
           nloc       = 0
           iloc[nloc] = icen
           jloc[nloc] = jcen
           latc       = ensmat.latitude.values[jcen]
           lonc       = ensmat.longitude.values[icen]

           if mask_land:
              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr2, self.config)
              vDict['flip_lon'] = self.config['model'].get('flip_lon','False')
              lmask = g1.read_static_field(self.config['metric'].get('static_fields_file'), 'landmask', vDict).values
           else:
              lmask      = np.ones(e_mean.shape)
              lmask[:,:] = 1.0

           k = 0
           while k <= nloc:

              for i in range(max(iloc[k]-1,0), min(iloc[k]+2,nlon)):
                 for j in range(max(jloc[k]-1,0), min(jloc[k]+2,nlat)):
                    if e_std[j,i] >= stdmax and lmask[j,i] >= lmaskmin and fmgrid[j,i] < 1.0:
                       nloc = nloc + 1
                       iloc[nloc] = i
                       jloc[nloc] = j
                       fmgrid[j,i] = 1.0

              k = k + 1

           fmout = np.zeros(g1.nens)
           npts  = np.sum(fmgrid)

           #  Average precipitation
           for n in range(g1.nens):
              fmout[n] = np.sum(fmgrid[:,:]*ensmat[n,:,:]) / npts

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(11,6.5), constrained_layout=True)

           colorlist = ("#FFFFFF", "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000", "#FFFF00", \
                        "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000", "#FF00FF", "#9955C9")

           plotBase = self.config.copy()
           plotBase['subplot']       = 'True'
           plotBase['subrows']       = 1
           plotBase['subcols']       = 2
           plotBase['subnumber']     = 1
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'None'
           plotBase['right_labels'] = 'None'
           plotBase['bottom_labels'] = 'None'
           ax1 = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the ensemble-mean precipitation on the left panel
           mpcp = [0.0, 0.25, 0.50, 1., 1.5, 2., 4., 6., 8., 12., 16., 24., 32., 64., 96., 97.]
           norm = matplotlib.colors.BoundaryNorm(mpcp,len(mpcp))
           pltf1 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_mean,mpcp, \
                                 cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max',transform=ccrs.PlateCarree())

           pltb = plt.contour(ensmat.longitude.values,ensmat.latitude.values,fmgrid,[0.5],linewidths=2.5,colors='w', \
                                    zorder=10,transform=ccrs.PlateCarree())
           if 'lonc' in locals():
              plt.plot(lonc, latc, '+', color='k', markersize=12, markeredgewidth=3, transform=ccrs.PlateCarree())

           cbar = plt.colorbar(pltf1, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mpcp)
           cbar.set_ticks(mpcp[1:(len(mpcp)-1):2])

           plt.title('Mean')

           plotBase['subnumber']     = 2
           ax2 = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the ensemble standard deviation precipitation on the right panel
           spcp = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 43.]
           norm = matplotlib.colors.BoundaryNorm(spcp,len(spcp))
           pltf2 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_std,spcp, \
                                 cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,fmgrid,[0.49, 0.51],linewidths=2.5, colors='w',transform=ccrs.PlateCarree()) 
           if 'lonc' in locals():
              plt.plot(lonc, latc, '+', color='k', markersize=12, markeredgewidth=3, transform=ccrs.PlateCarree())

           cbar = plt.colorbar(pltf2, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=spcp)
           cbar.set_ticks(spcp[1:(len(spcp)-1)])

           plt.title('Standard Deviation')

           fig.suptitle('F{0}-F{1} Precipitation ({2}-{3})'.format('%0.3i' % fhr1, '%0.3i' % fhr2, date1_str, date2_str), fontsize=16)

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],'%0.3i' % fhr2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'precipitation', 'FORECAST_METRIC_SHORT_NAME': 'pcp', \
                      'FORECAST_HOUR1': int(fhr1), 'FORECAST_HOUR2': int(fhr2), 'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, \
                      'LONGITUDE2': lon2, 'LAND_MASK_MINIMUM': lmaskmin, 'ADAPT': str(adapt), 'TIME_ADAPT': str(time_adapt), \
                      'TIME_ADAPT_DOMAIN': time_dbuff, 'TIME_ADAPT_FREQ': time_freq, 'ADAPT_PCP_MIN': pcpmin}

           if 'lonc' in locals():
              fmetatt.update({'LATITUDE_CENTER': latc, 'LONGITUDE_CENTER': lonc})

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'mm', 'description': 'precipitation ensemble mean'}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['metric_domain'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': '', 'description': ' precipitation metric domain'}, 'data': fmgrid.data}
           endict['metric_domain'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': 'mm', 'description': 'precipitation'}, 'data': fmout}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr2,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr2, metname))

           del f_met


    def __precipitation_eof(self):
        '''
        Function that computes precipitation EOF metric, which is calculated by taking the EOF of 
        the ensemble precipitation forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean precipitation pattern 
        along with the precipitation perturbation that is consistent with the first EOF. 
        '''

        logging.warning('  Precipitation EOF Metrics:')

        if eval(self.config['metric'].get('metric_from_fields','False')):

           for pfile in glob.glob('{0}/*pcp1.nc'.format(self.config['locations']['work_dir'])):
              self.metlist.append('{0}_{1}'.format(pfile.split('_')[-2], pfile.split('_')[-1].split('.')[-2]))
           return


        for infile in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('precip_metric_loc'),self.datea_str)):

           fint = int(self.config['metric'].get('fcst_int',self.config['model']['fcst_hour_int']))

           try:
              conf = configparser.ConfigParser()
              conf.read(infile)
              fhr1 = int(conf['definition'].get('forecast_hour1',self.config['metric'].get('precip_eof_forecast_hour1','48')))
              fhr2 = int(conf['definition'].get('forecast_hour2',self.config['metric'].get('precip_eof_forecast_hour2','120')))
              lat1 = float(conf['definition'].get('latitude_min',32))
              lat2 = float(conf['definition'].get('latitude_max',50))
              lon1 = float(conf['definition'].get('longitude_min',-125))
              lon2 = float(conf['definition'].get('longitude_max',-115))
              adapt = eval(conf['definition'].get('adapt',self.config['metric'].get('precip_eof_adapt','False')))
              time_adapt = eval(conf['definition'].get('time_adapt',self.config['metric'].get('precip_eof_time_adapt','False')))
              time_dbuff = float(conf['definition'].get('time_adapt_domain',self.config['metric'].get('precip_eof_time_adapt_domain',2.0)))
              time_freq = int(conf['definition'].get('time_adapt_freq',self.config['metric'].get('precip_eof_time_adapt_freq',6)))
              pcpmin = float(conf['definition'].get('adapt_pcp_min',self.config['metric'].get('precip_eof_adapt_pcp_min','12.7')))
              lmaskmin = float(conf['definition'].get('land_mask_minimum',self.config['metric'].get('land_mask_minimum','0.2')))
              mask_land = eval(conf['definition'].get('land_mask',self.config['metric'].get('precip_eof_land_mask','False')))
              frozen = eval(conf['definition'].get('frozen_mask',self.config['metric'].get('precip_eof_frozen_mask','False')))
              metname = conf['definition'].get('metric_name','pcpeof')
              eofn = int(conf['definition'].get('eof_number',1))
           except:
              logging.warning('  {0} does not exist.  Using parameter and/or default values'.format(infile))

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr2, self.config)

           #  Skip frozen precipitation for GEFS for now.  Can be removed when this is solved.
           if frozen and not (g1.has_precip_category or g1.has_frozen_fraction):
              logging.warning('  This model does not support frozen precipitation metrics.  Skipping {0} metric'.format(metname))
              continue

           #  Now figure out the 24 h after landfall, so we can set the appropriate 24 h period.
           if time_adapt:

              vDict = {'latitude': (lat1-0.00001, lat2+0.00001), 'longitude': (lon1-0.00001, lon2+0.00001),
                       'description': 'precipitation', 'units': 'mm', '_FillValue': -9999., 'flip_lon': self.config['model'].get('flip_lon','False')}
              vDict = g1.set_var_bounds('precipitation', vDict)
              lmask = g1.read_static_field(self.config['metric'].get('static_fields_file'), 'landmask', vDict).values

              #  Read precipitation over the default window, calculate SD, search for maximum value
              ensmat = self.__read_precip(fhr1, fhr2, self.config, vDict)

              if frozen:

                 ensmat[:,:,:] = 0.
                 fint = int(self.config['metric'].get('fcst_int',6))
                 for fhr in range(fhr1+fint, fhr2+fint, fint):
                    pcp = self.__read_precip(fhr-fint, fhr, self.config, vDict)
                    g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

                    if g1.has_precip_category:
                       for n in range(g1.nens):
                          ensmat[n,:,:] = ensmat[n,:,:] + np.where(np.squeeze(g1.read_grib_field('precip_type', n, vDict)) >= 2.0, \
                                                                   np.squeeze(pcp[n,:,:]), 0.0)
                    elif g1.has_frozen_fraction:
                       for n in range(g1.nens):
                          frfrac = g1.read_grib_field('snow_fraction', n, vDict) + g1.read_grib_field('freeze_rain_fraction', n, vDict)
                          ensmat[n,:,:] = ensmat[n,:,:] + frfrac[:,:] * np.squeeze(pcp[n,:,:])


              e_std = np.std(ensmat, axis=0)
              estd_mask = e_std.values[:,:] * lmask[:,:]

              maxloc = np.where(estd_mask == estd_mask.max())
              lonc   = ensmat.longitude.values[int(maxloc[1])]
              latc   = ensmat.latitude.values[int(maxloc[0])]

              logging.info('    Precip. Time Adapt: Lat/Lon center: {0}, {1}'.format(latc,lonc))

              pmax = -1.0
              for fhr in range(fhr1, fhr2-24+time_freq, time_freq):

                 psum = np.sum(np.mean(self.__read_precip(fhr, fhr+24, self.config, vDict), axis=0))
                 logging.info('    Precip. Time Adapt: {0}-{1} h, area precip: {2}'.format(fhr,fhr+24,psum.values))
                 if psum > pmax:
                    fhr1 = fhr
                    fhr2 = fhr+24
                    pmax = psum

           logging.warning('  Precipitation Metric ({0}), Hours: {1}-{2}, Lat: {3}-{4}, Lon: {5}-{6}'.format(metname,fhr1,fhr2,lat1,lat2,lon1,lon2))


           #  Read the total precipitation, scale to a 24 h value 
           vDict = {'latitude': (lat1-0.00001, lat2+0.00001), 'longitude': (lon1-0.00001, lon2+0.00001),
                       'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
           ensmat = self.__read_precip(fhr1, fhr2, self.config, vDict)
           ensmat[:,:,:] = ensmat[:,:,:] * 24. / float(fhr2-fhr1)

           if frozen:

              ensmat[:,:,:] = 0.
              fint = int(self.config['metric'].get('fcst_int',6))
              for fhr in range(fhr1+fint, fhr2+fint, fint):
                 pcp = self.__read_precip(fhr-fint, fhr, self.config, vDict)
                 g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)
                    
                 if g1.has_precip_category:
                    for n in range(g1.nens):
                       ensmat[n,:,:] = ensmat[n,:,:] + np.where(np.squeeze(g1.read_grib_field('precip_type', n, vDict)) >= 2.0, \
                                                                   np.squeeze(pcp[n,:,:]), 0.0)
                 elif g1.has_frozen_fraction:
                    for n in range(g1.nens):
                       frfrac = g1.read_grib_field('snow_fraction', n, vDict) + g1.read_grib_field('freeze_rain_fraction', n, vDict)
                       ensmat[n,:,:] = ensmat[n,:,:] + frfrac[:,:] * np.squeeze(pcp[n,:,:])

              ensmat[:,:,:] = ensmat[:,:,:] * 24. / float(fhr2-fhr1)

           e_mean = np.mean(ensmat, axis=0)
           e_std  = np.std(ensmat, axis=0)
           ensmat = ensmat - e_mean
           nens   = len(ensmat[:,0,0])

           if mask_land:
              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr2, self.config)
              vDict['flip_lon'] = self.config['model'].get('flip_lon','False')
              lmask = g1.read_static_field(self.config['metric'].get('static_fields_file'), 'landmask', vDict).values
           else:
              lmask      = np.ones(e_mean.shape)
              lmask[:,:] = 1.0

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config['model'].get('grid_type','LatLon') == 'LatLon':
              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)       
              wgts = np.sqrt(coslat)[..., np.newaxis]

           #  Find the location of precipitation max. and then the area over which this exceeds a certain threshold
           if adapt:

              nlon  = len(e_mean.longitude.values)
              nlat  = len(e_mean.latitude.values)

              # Search for maximum in ensemble precipitation SD 
              if np.amax(lmask) < lmaskmin:
                 logging.error('  precipitation metric does not have any land points.  Skipping metric.')
                 continue

              estd_mask = e_mean.values[:,:] * lmask[:,:]
#              estd_mask = e_std.values[:,:] * lmask.values[:,:]

              stdmax = estd_mask.max()
              maxloc = np.where(estd_mask == stdmax)
              icen   = int(maxloc[1])
              jcen   = int(maxloc[0])
              lonc   = e_mean.longitude.values[icen]
              latc   = e_mean.latitude.values[jcen]

              fmgrid = e_mean.copy()
              fmgrid[:,:] = 0.0
              fmgrid[jcen,icen] = 1.0

              iloc       = np.zeros(nlon*nlat, dtype=int)
              jloc       = np.zeros(nlon*nlat, dtype=int)
              nloc       = 0
              iloc[nloc] = icen
              jloc[nloc] = jcen

              k = 0 
              while k <= nloc: 

                 for i in range(max(iloc[k]-1,0), min(iloc[k]+2,nlon)):
                    for j in range(max(jloc[k]-1,0), min(jloc[k]+2,nlat)):
                       if e_mean[j,i] >= pcpmin and lmask[j,i] >= lmaskmin and fmgrid[j,i] < 1.0:
                          nloc = nloc + 1
                          iloc[nloc] = i
                          jloc[nloc] = j
                          fmgrid[j,i] = 1.0     

                 k = k + 1

              #  Evaluate whether the forecast metric grid has enough land points
              if np.sum(fmgrid) <= 1.0:
                 logging.error('  precipitation metric does not have any land points after doing search.  Skipping metric.')
                 continue

              #  Find the grid bounds for the precipitation domain (for plotting purposes)
              i1 = nlon-1
              i2 = 0
              j1 = nlat-1
              j2 = 0
              for i in range(nlon):
                 for j in range(nlat):
                    if fmgrid[j,i] > 0.0:
                       i1 = np.minimum(i,i1)
                       i2 = np.maximum(i,i2)
                       j1 = np.minimum(j,j1)
                       j2 = np.maximum(j,j2)

              i1 = np.maximum(i1-5,0)
              i2 = np.minimum(i2+5,nlon-1)
              j1 = np.maximum(j1-5,0)
              j2 = np.minimum(j2+5,nlat-1)

              ngrid = -1
              ensarr = xr.DataArray(name='ensemble_data', data=np.zeros([nens, nlat*nlon]), \
                                      dims=['time', 'state'])

              for i in range(nlon):
                 for j in range(nlat):
                    if fmgrid[j,i] > 0.0:
                       ngrid = ngrid + 1
                       ensarr[:,ngrid] = ensmat[:,j,i].data * np.sqrt(coslat[j])

              solver = Eof_xarray(ensarr[:,0:ngrid])

              #  Restrict domain for plotting purposes
              lon1 = ensmat.longitude.values[i1]
              lon2 = ensmat.longitude.values[i2]
              lat1 = ensmat.latitude.values[j1]
              lat2 = ensmat.latitude.values[j2]

              ensmat = ensmat.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
              fmgrid = fmgrid.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
              e_mean = e_mean.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))

           else:

              if mask_land:

                 nlat = len(ensmat[0,:,0])
                 nlon = len(ensmat[0,0,:])
                 ngrid = -1
                 ensarr = xr.DataArray(name='ensemble_data', data=np.zeros([nens, nlat*nlon]), \
                                         dims=['time', 'state'])

                 if self.config['model'].get('grid_type','LatLon') == 'LatLon':

                    for i in range(nlon):
                       for j in range(nlat):
                          if lmask[j,i] > 0.0:
                             ngrid = ngrid + 1
                             ensarr[:,ngrid] = ensmat[:,j,i].data * np.sqrt(coslat[j]) * lmask[j,i]

                 else:

                    for i in range(nlon):
                       for j in range(nlat):
                          if lmask[j,i] > 0.0:
                             ngrid = ngrid + 1
                             ensarr[:,ngrid] = ensmat[:,j,i].data * lmask[j,i]
           
                 solver = Eof_xarray(ensarr[:,0:ngrid])


              else:

                 #  Compute the EOF of the precipitation pattern and then the PCs
                 if self.config['model'].get('grid_type','LatLon') == 'LatLon':

                    coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
                    wgts = np.sqrt(coslat)[..., np.newaxis]
                    solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

                 else:

                    solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))


           pcout  = solver.pcs(npcs=3, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the precipitation pattern associated with a 1 PC perturbation
           dpcp = np.zeros(e_mean.shape)

           for n in range(nens):
              dpcp[:,:] = dpcp[:,:] + ensmat[n,:,:] * pc1[n]

           dpcp[:,:] = dpcp[:,:] / float(nens)

           if np.sum(dpcp) < 0.0:
              dpcp[:,:] = -dpcp[:,:]
              pc1[:]    = -pc1[:]

           if self.config['metric'].get('precipitation_eof_flip', 'False') == 'True':
              dpcp[:,:] = -dpcp[:,:]
              pc1[:]    = -pc1[:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#FFFFFF", "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000", "#FFFF00", \
                        "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000", "#FF00FF", "#9955C9")

           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Add the ensemble-mean precipitation in shading
           mpcp = [0.0, 0.25, 0.50, 1., 1.5, 2., 4., 6., 8., 12., 16., 24., 32., 64., 96., 97.]
           norm = matplotlib.colors.BoundaryNorm(mpcp,len(mpcp))
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_mean,mpcp, transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           #  Add contours of the precipitation EOF
           pcpfac = np.ceil(np.max(abs(dpcp)) / 5.0)
           cntrs = np.array([-5., -4., -3., -2., -1., 1., 2., 3., 4., 5]) * pcpfac
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,dpcp,cntrs,linewidths=1.5, \
                                colors='k', zorder=10, transform=ccrs.PlateCarree())

           if adapt:
              pltb = plt.contour(ensmat.longitude.values,ensmat.latitude.values,fmgrid,[0.5],linewidths=2.5, colors='0.4', zorder=10)
              if 'lonc' in locals():
                 plt.plot(lonc, latc, '+', color='k', markersize=12, markeredgewidth=3, transform=ccrs.PlateCarree())

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mpcp)
           cbar.set_ticks(mpcp[1:(len(mpcp)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = solver.varianceFraction(neigs=1).data
           else:
              fracvar = solver.varianceFraction(neigs=eofn)[-1].data
           plt.title("{0} {1}-{2} hour Precipitation, {3} of variance".format(str(self.datea_str),fhr1,fhr2,'%4.3f' % fracvar))

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],'%0.3i' % fhr2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'precipitation PC', 'FORECAST_METRIC_SHORT_NAME': 'pcpeof', \
                      'FORECAST_HOUR1': int(fhr1), 'FORECAST_HOUR2': int(fhr2), 'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, \
                      'LONGITUDE2': lon2, 'LAND_MASK_MINIMUM': lmaskmin, 'ADAPT': str(adapt), 'TIME_ADAPT': str(time_adapt), \
                      'TIME_ADAPT_DOMAIN': time_dbuff, 'TIME_ADAPT_FREQ': time_freq, 'ADAPT_PCP_MIN': pcpmin, 'EOF_NUMBER': int(eofn), 'VAR_FRACTION': fracvar}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'mm', 'description': 'precipitation ensemble mean'}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'mm', 'description': 'precipitation EOF pattern'}, 'data': dpcp}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'precipitation PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr2,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr2, metname))

           del f_met


    def __read_precip(self, fhr1, fhr2, confgrib, vDict):

        g2 = self.dpp.ReadGribFiles(self.datea_str, fhr2, confgrib)
        vDict = g2.set_var_bounds('precipitation', vDict)
        ensmat = g2.create_ens_array('precipitation', g2.nens, vDict)

        #  Calculate total precipitation for models that provide total precipitation over model run
        if g2.has_total_precip:

           if fhr1 > 0:
              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1, confgrib)
              for n in range(g2.nens):
                 ens1 = np.squeeze(g1.read_grib_field('precipitation', n, vDict))
                 ens2 = np.squeeze(g2.read_grib_field('precipitation', n, vDict))
                 ensmat[n,:,:] = ens2[:,:] - ens1[:,:]
           else:
              for n in range(g2.nens):
                 ensmat[n,:,:] = np.squeeze(g2.read_grib_field('precipitation', n, vDict))

           if hasattr(ens2, 'units'):
              if ens2.units == "m":
                 vscale = 1000.
              else:
                 vscale = 1.
           else:
              vscale = 1.

        #  Calculate total precipitaiton for models that output precipitation in time periods
        else:

           fint = int(self.config['metric'].get('fcst_int',6))
           for fhr in range(fhr1+fint, fhr2+fint, fint):
              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, confgrib)
              for n in range(g1.nens):
                 ensmat[n,:,:] = ensmat[n,:,:] + np.squeeze(g1.read_grib_field('precipitation', n, vDict))

           if hasattr(g1.read_grib_field('precipitation', 0, vDict), 'units'):
              if g1.read_grib_field('precipitation', 0, vDict).units == "m":
                 vscale = 1000.
              else:
                 vscale = 1.
           else:
              vscale = 1.

        return ensmat[:,:,:] * vscale


    def __precip_basin_eof(self):

        #wget --no-check-certificate https://cw3e.ucsd.edu/Projects/QPF/data/eps_watershed_precip8.nc

        if not os.path.isfile('watershed_precip.nc'):
           logging.warning('  {0} is not present.  Exiting.'.format('watershed_precip.nc'))
           return None

        db = pd.read_csv(filepath_or_buffer=self.config['metric'].get('basin_huc_file'), \
                           sep = ',', header=None, skipinitialspace=True, quotechar="\"")
        db.columns = ['ID','Name','Size']

        ds = xr.open_dataset('watershed_precip.nc', decode_times=False).rename({'time': 'hour'}) 
        if ds.attrs['init'] != self.datea_str:
           logging.warning('  {0} init date ({1}) does not match the forecast.  Exiting and deleting file'.format('watershed_precip.nc',ds.attrs['init']))
           os.remove('watershed_precip.nc')
           return None

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('basin_metric_loc'),self.datea_str)):

           try:
              f = open(infull, 'r')
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute basin precip EOF'.format(infull))
              return None

           print(infull)

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr1 = int(conf['definition'].get('forecast_hour1',0))
              fhr2 = int(conf['definition'].get('forecast_hour2',120))
              metname = conf['definition'].get('metric_name','basin')
              eofn = int(conf['definition'].get('eof_number',1))
              basin_input = conf['definition'].get('basin_name')
              hucid_input = conf['definition'].get('hucid')
              auto_domain = eval(conf['definition'].get('automated','False'))
              auto_sdmin  = float(conf['definition'].get('automated_sd_min',0.80))
              accumulated = eval(conf['definition'].get('accumulation','False'))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute precip EOF'.format(infull))
              return None

           if auto_domain:

              mean_min = 12.7
              bmea = np.mean(np.sum(ds.precip.sel(hour=slice(fhr1, fhr2)).squeeze().load(), axis=0), axis=0)
              bstd = np.std(np.sum(ds.precip.sel(hour=slice(fhr1, fhr2)).squeeze().load(), axis=0), axis=0)

              brat = bstd[:] / np.fmax(bmea, mean_min)
#              imax = np.argmax(brat.values)
              imax = np.argmax(bstd.values)

              print(bmea[imax].values,bstd[imax].values,brat[imax].values)

              if bmea[imax] < mean_min:
                 logging.error('  basin precipitation metric center point is below minimum.  Skipping metric.')
                 continue

              hucid_list = []
              basin_list = []
              index_list = []

              for basin in range(len(brat)):
                 if bstd[basin] >= auto_sdmin * bstd[imax]:
#                 if brat[basin] >= auto_sdmin * brat[imax]:
                    hucid = ds.HUCID[basin]
                    hucid_list.append(int(hucid))
                    basin_list.append(db[db['ID'] == int(hucid)]['Name'].values)
                    index_list.append(basin)
#                    print('  adding basin',basin_list[-1],bstd[basin].values)

           else:

              if hucid_input:

                 hucid_list = [e.strip() for e in hucid_input.split(',')]

                 basin_list = []
                 for hucid in hucid_list:
                    basin_list.append(db[db['ID'] == int(hucid)]['Name'].values)

              else:

                 basin_list = [e.strip() for e in basin_input.split(',')]

                 hucid_list = []
                 for basin in basin_list:
                    hucid_list.append(db[db['Name'] == basin]['ID'].values[0])

              index_list = []
              for hucid in hucid_list:
                 index_list.append(db[db['ID'] == int(hucid)].index.values[0])


           basinsum = 0.
 
           ensmat = ((ds.precip.sel(hour=slice(fhr1, fhr2), HUCID=int(hucid_list[0])).squeeze()).transpose()).load()

           for hucid in hucid_list:

              bmask = db['ID'] == int(hucid)
              logging.debug('{0} ({1}), {2} Acres'.format(db[bmask]['Name'].values[0],db[bmask]['ID'].values[0],np.round(db[bmask]['Size'].values[0])))

              prate = ((ds.precip.sel(hour=slice(fhr1, fhr2), HUCID=int(hucid)).squeeze()).transpose()).load()
              prate[:,:] = prate[:,:] * db[bmask]['Size'].values
              basinsum = basinsum + db[bmask]['Size'].values

#              ensmat = prate.copy(deep=True)
              if accumulated:

                 ylabel = 'Accumulated Precipitation (mm)'
                 for t in range(ensmat.shape[1]):
                    ensmat[:,t] = ensmat[:,t] + np.sum(prate[:,0:t],axis=1)

              else:

                 ensmat[:,:] = ensmat[:,:] + prate[:,:]
                 ylabel = 'Precipitation Rate (mm 6 $\\mathregular{h^{-1}}$)'

           
           ensmat[:,:] = ensmat[:,:] / basinsum 
           e_mean = np.mean(ensmat, axis=0)
           for n in range(ensmat.shape[0]):
              ensmat[n,:] = ensmat[n,:] - e_mean[:]

           solver = Eof_xarray(ensmat.rename({'Ensemble': 'time'}))
           pcout  = solver.pcs(npcs=3, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the precipitation pattern associated with a 1 PC perturbation
           dpcp = np.zeros(e_mean.shape)

           for n in range(ensmat.shape[0]):
              dpcp[:] = dpcp[:] + ensmat[n,:] * pc1[n]

           dpcp[:] = dpcp[:] / float(ensmat.shape[0])

           if np.sum(dpcp) < 0.0:
              dpcp[:] = -dpcp[:]
              pc1[:]    = -pc1[:]

           #  Create plots of MSLP and maximum wind for each member, mean and EOF perturbation
           fig = plt.figure(figsize=(13, 7.5))

           gdf = gpd.read_file(self.config['metric'].get('basin_shape_file'))

           plotBase = self.config.copy()
           plotBase['subplot']       = 'True'
           plotBase['subrows']       = 1
           plotBase['subcols']       = 2
           plotBase['subnumber']     = 1
           plotBase['grid_interval'] = 180
           plotBase['left_labels'] = 'None'
           plotBase['right_labels'] = 'None'
           plotBase['bottom_labels'] = 'None'
           ax0 = background_map('PlateCarree', -126, -105, 30, 53, plotBase)

           gdf.plot(ax=ax0, color='white', edgecolor='silver', linewidth=0.5)
           for basin in index_list:
              gdf.iloc[[basin]].plot(ax=ax0, facecolor='gold')

           if auto_domain:
              gdf.iloc[[imax]].plot(ax=ax0, facecolor='red')

           ax1  = fig.add_axes([0.54, 0.16, 0.42, 0.67])

           for n in range(ensmat.shape[0]):
              ax1.plot(ensmat.hour, ensmat[n,:]+e_mean[:], color='lightgray')

           ax1.plot(ensmat.hour, e_mean, color='black', linewidth=3)
           ax1.plot(ensmat.hour, e_mean[:]+dpcp[:], '--', color='black', linewidth=3)

           ax1.set_xlim((fhr1, fhr2))

           init   = dt.datetime.strptime(self.datea_str, '%Y%m%d%H')
           ticklist = []
           for fhr in np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.):

              datef  = init + dt.timedelta(hours=fhr)
              ticklist.append(datef.strftime("%HZ\n%m/%d"))

           ax1.set_xticks(np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.))
           ax1.set_xticklabels(ticklist, fontsize=12)           

           ax1.set_ylabel(ylabel, fontsize=15)
           ax1.set_ylim(bottom=0.)
           ax1.tick_params(axis='y', labelsize=12)

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.suptitle("{0} {1}-{2} hour Precipitation, {3} of variance".format(str(self.datea_str),fhr1,fhr2,\
                                  fracvar), x=0.5, y=0.86, fontsize=15)

           init   = dt.datetime.strptime(self.datea_str, '%Y%m%d%H')
           ticklist = []
           for fhr in np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.):

              datef  = init + dt.timedelta(hours=fhr)
              ticklist.append(datef.strftime("%HZ\n%m/\n%d"))

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],'%0.3i' % fhr2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=150, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'river basin precipitation PC', 'FORECAST_METRIC_SHORT_NAME': 'rpcpeof', \
                      'FORECAST_HOUR1': int(fhr1), 'FORECAST_HOUR2': int(fhr2), 'AUTOMATED': str(auto_domain), \
                      'AUTOMATED_SD_MIN': auto_sdmin, 'ACCUMULATION': str(accumulated), 'EOF_NUMBER': int(eofn)} 

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': ensmat.shape[0]}, 'data_vars': {}}
           f_met['coords']['forecast_hour'] = {'dims': ('forecast_hour'), 'attrs': {'units': 'hr', 'description': 'forecast hour'}, 'data': ensmat.hour.values}
           endict['forecast_hour'] = {'dtype': 'float32'}
           f_met['coords']['huc_id'] = {'dims': ('huc_id',), 'attrs': {'description': 'HUC ID'}, 'data': hucid_list}
           endict['huc_id'] = {'dtype': 'int'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('forecast_hour'), 'attrs': {'units': 'mm', 'description': 'basin precipitation ensemble mean'}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('forecast_hour'), 'attrs': {'units': 'mm', 'description': 'basin precipitation EOF pattern'}, 'data': dpcp}
           endict['EOF_pattern'] = {'dtype': 'float32'}

           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'precipitation PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr2,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr2, metname))


    def __slp_eof(self):
        '''
        Function that computes SLP EOF metric, which is calculated by taking the EOF of 
        the ensemble SLP forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean SLP pattern 
        along with the SLP perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('slp_metric_loc'),self.datea_str)):

           try:
              f = open(infull, 'r')
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute SLP EOF'.format(infull))
              return None

           print(infull)

           #  Read the text file that contains information on the SLP metric
           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr = int(conf['definition'].get('forecast_hour'))
              metname = conf['definition'].get('metric_name','mslpeof')
              eofn = int(conf['definition'].get('eof_number',1))
              lat1 = float(conf['definition'].get('latitude_min'))
              lat2 = float(conf['definition'].get('latitude_max'))
              lon1 = float(conf['definition'].get('longitude_min'))
              lon2 = float(conf['definition'].get('longitude_max'))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute SLP EOF'.format(infull))
              return None

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'sea-level pressure', 'units': 'hPa', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ensmat = g1.create_ens_array('sea_level_pressure', g1.nens, vDict)

           #  Read the ensemble SLP fields, compute the mean
           for n in range(g1.nens):
              ensmat[n,:,:] = g1.read_grib_field('sea_level_pressure', n, vDict)

           if g1.read_grib_field('sea_level_pressure', 0, vDict).units != 'hPa':
              ensmat[:,:,:] = ensmat[:,:,:] * 0.01

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config['model'].get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              wgts = np.sqrt(coslat)[..., np.newaxis]
              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

           else:

              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))

           pcout  = np.squeeze(solver.pcs(npcs=3, pcscaling=1))
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the SLP pattern associated with a 1 PC perturbation
           dslp = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              dslp[:,:] = dslp[:,:] + ensmat[n,:,:] * pc1[n]

           dslp[:,:] = dslp[:,:] / float(g1.nens)

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#9A32CD","#00008B","#3A5FCD","#00BFFF","#B0E2FF","#FFFFFF","#FFEC8B","#FFA500","#FF4500","#B22222","#FF82AB")

           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the SLP EOF pattern in shading
           slpfac = np.ceil(np.max(np.abs(dslp)) / 5.0)
           cntrs = np.array([-5., -4., -3., -2., -1., 1., 2., 3., 4., 5]) * slpfac
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,dslp,cntrs,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), extend='both')

           #  Plot the ensemble-mean SLP field in contours
           mslp = [948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040]
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,e_mean,mslp, \
                               linewidths=1.5,colors='k',zorder=10,transform=ccrs.PlateCarree())

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=cntrs)
           cbar.set_ticks(cntrs[1:(len(cntrs)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.title("{0} {1} hour SLP, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           #  Create a output directory with the metric file
           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': '', 'FORECAST_METRIC_NAME': 'SLP PC', 'FORECAST_METRIC_SHORT_NAME': 'slpeof', 'FORECAST_HOUR': int(fhr), \
                      'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, 'LONGITUDE2': lon2, 'EOF_NUMBER': int(eofn)}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'hPa', 'description': 'sea-level pressure ensemble mean'}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'hPa', 'description': 'sea-level pressure EOF pattern'}, 'data': dslp}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'sea-level pressure PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr, metname))


    def __hght_eof(self):
        '''
        Function that computes SLP EOF metric, which is calculated by taking the EOF of 
        the ensemble SLP forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean SLP pattern 
        along with the SLP perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('hght_metric_loc'),self.datea_str)):

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr   = int(conf['definition'].get('forecast_hour'))
              lat1  = float(conf['definition'].get('latitude_min'))
              lat2  = float(conf['definition'].get('latitude_max'))
              lon1  = float(conf['definition'].get('longitude_min'))
              lon2  = float(conf['definition'].get('longitude_max'))
              eofn  = int(conf['definition'].get('eof_number',1))
              level = float(conf['definition'].get('pressure', 500)) 
              metname = conf['definition'].get('metric_name','h{0}hPa'.format(int(level)))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(infull))
              return None

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           inpath, infile = infull.rsplit('/', 1)
#           infile1, metname = infile.split('_', 1)
           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                    'description': 'Height', 'units': 'm', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ensmat = g1.create_ens_array('geopotential_height', g1.nens, vDict)

           #  Read the ensemble SLP fields, compute the mean
           for n in range(g1.nens):
              ensmat[n,:,:] = np.squeeze(g1.read_grib_field('geopotential_height', n, vDict))

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config['model'].get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              wgts = np.sqrt(coslat)[..., np.newaxis]
              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

           else:

              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))

           pcout  = np.squeeze(solver.pcs(npcs=3, pcscaling=1))
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the SLP pattern associated with a 1 PC perturbation
           dhght = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              dhght[:,:] = dhght[:,:] + ensmat[n,:,:] * pc1[n]

           dhght[:,:] = dhght[:,:] / float(g1.nens)

           if np.sum(dhght) < 0.:
              pc1[:]     = -pc1[:]
              dhght[:,:] = -dhght[:,:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#9A32CD","#00008B","#3A5FCD","#00BFFF","#B0E2FF","#FFFFFF","#FFEC8B","#FFA500","#FF4500","#B22222","#FF82AB")

           ax = plt.axes(projection=ccrs.PlateCarree())
           ax = self.__background_map(ax, lat1, lon1, lat2, lon2)

           #  Plot the SLP EOF pattern in shading
           hfac = np.ceil(np.max(dhght) / 5.0)
           cntrs = np.array([-5., -4., -3., -2., -1., 1., 2., 3., 4., 5]) * hfac
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,dhght,cntrs,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), extend='both')

           #  Plot the ensemble-mean SLP field in contours
           if level == 700:
              mhght = np.arange(2400, 3300, 20)
           elif level == 500:
              mhght = np.arange(4800, 6000, 30)
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,e_mean,mhght,\
                               transform=ccrs.PlateCarree(),linewidths=1.5,colors='k',zorder=10)

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=cntrs)
           cbar.set_ticks(cntrs[1:(len(cntrs)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.title("{0} {1} hour height, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': level, 'FORECAST_METRIC_NAME': 'Height PC', 'FORECAST_METRIC_SHORT_NAME': 'hghteof', 'FORECAST_HOUR': int(fhr), \
                      'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, 'LONGITUDE2': lon2, 'PRESSURE': level, 'EOF_NUMBER': int(eofn)}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'm', 'description': '{0} hPa height ensemble mean'.format(level)}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'm', 'description': '{0} hPa height EOF pattern'.format(level)}, 'data': dhght}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'height PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr, metname))


    def __pvort_eof(self):
        '''
        Function that computes SLP EOF metric, which is calculated by taking the EOF of 
        the ensemble SLP forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean SLP pattern 
        along with the SLP perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('pvort_metric_loc'),self.datea_str)):

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr   = int(conf['definition'].get('forecast_hour'))
              lat1  = float(conf['definition'].get('latitude_min'))
              lat2  = float(conf['definition'].get('latitude_max'))
              lon1  = float(conf['definition'].get('longitude_min'))
              lon2  = float(conf['definition'].get('longitude_max'))
              eofn  = int(conf['definition'].get('eof_number',1))
              level = float(conf['definition'].get('pressure', 250))
              metname = conf['definition'].get('metric_name','pv{0}hPa'.format(int(level)))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute PV EOF'.format(infull))
              return None

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           inpath, infile = infull.rsplit('/', 1)
           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           pvec = g1.read_pressure_levels('temperature')
           idx  = np.where(pvec==level)
           lev1 = np.min(pvec[(int(idx[0])-1):(int(idx[0])+2)])
           lev2 = np.max(pvec[(int(idx[0])-1):(int(idx[0])+2)])

           vDict = g1.set_var_bounds('zonal_wind', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (lev1, lev2), \
                                                    'description': '{0} hPa Potential Vorticity'.format(level), 'units': 'PVU', '_FillValue': -9999.})
           tDict = g1.set_var_bounds('temperature', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (lev1, lev2), \
                                                     'description': '{0} hPa Potential Vorticity'.format(level), 'units': 'PVU', '_FillValue': -9999.})

           ensmat = g1.create_ens_array('zonal_wind', g1.nens, vDict)
           lats = ensmat.latitude.values * units('degrees')
           lons = ensmat.longitude.values * units('degrees')

           dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats, x_dim=-1, y_dim=-2, geod=None)
           dx = np.maximum(dx, 1.0 * units('m'))

           for n in range(g1.nens):

              #  Read all the necessary files from file, smooth fields, so sensitivities are useful
              tmpk = g1.read_grib_field('temperature', n, tDict) * units('K')
              pres = tmpk.isobaricInhPa.values * units('hPa')
              thta = mpcalc.potential_temperature(pres[:, None, None], tmpk)

              uwnd = g1.read_grib_field('zonal_wind', n, vDict) * units('m/s')
              vwnd = g1.read_grib_field('meridional_wind', n, vDict) * units('m/s')

              #  Compute PV and place in ensemble array
              if self.config['model'].get('grid_type','LatLon') == 'LatLon':

                 pvout = np.abs(mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                                dx[None, :, :], dy[None, :, :], lats[None, :, None]))

                 ensmat[n,:,:] = grid_calc.calc_circ_llgrid(np.squeeze(pvout[np.where(pres == level * units('hPa'))[0],:,:]), \
                                                             300., lats, lons, eval(self.config['fields'].get('global','False')), len(lons), len(lats)) * 1.0e6

              else:

                 pvout = np.abs(mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                                dx[None, :, :], dy[None, :, :], lats[None, :, :]))

                 ensmat[n,:,:] = grid_calc.calc_circ(np.squeeze(pvout[np.where(pres == level * units('hPa'))[0],:,:]), \
                                                            300000., g1.dx, len(lats[0,:]), len(lats[:,0])) * 1.0e6

              del pres,thta,uwnd,vwnd,pvout

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config['model'].get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              wgts = np.sqrt(coslat)[..., np.newaxis]
              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

           else:

              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))

           pcout  = np.squeeze(solver.pcs(npcs=3, pcscaling=1))
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the SLP pattern associated with a 1 PC perturbation
           dpvort = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              dpvort[:,:] = dpvort[:,:] + ensmat[n,:,:] * pc1[n]

           dpvort[:,:] = dpvort[:,:] / float(g1.nens)

           if np.sum(dpvort) < 0.:
              pc1[:]      = -pc1[:]
              dpvort[:,:] = -dpvort[:,:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#9A32CD","#00008B","#3A5FCD","#00BFFF","#B0E2FF","#FFFFFF","#FFEC8B","#FFA500","#FF4500","#B22222","#FF82AB")

           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the SLP EOF pattern in shading
           hfac = np.ceil((np.max(dpvort) / 5.0)*10.) / 10.
           cntrs = np.array([-5., -4., -3., -2., -1., 1., 2., 3., 4., 5]) * hfac
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,dpvort,cntrs,transform=ccrs.PlateCarree(), \
                                   cmap=matplotlib.colors.ListedColormap(colorlist), extend='both')
           print('max',np.max(dpvort),hfac)

           #  Plot the ensemble-mean PV field in contours
           if level <= 500.:
              mpvort = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
           else:
              mpvort = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4]
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,e_mean,mpvort,\
                              transform=ccrs.PlateCarree(),linewidths=1.5,colors='k',zorder=10)

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=cntrs)
           cbar.set_ticks(cntrs[1:(len(cntrs)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.title("{0} {1} hour PV, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': level, 'FORECAST_METRIC_NAME': 'Potential Vorticity PC', 'FORECAST_METRIC_SHORT_NAME': 'pveof', 'FORECAST_HOUR': int(fhr), \
                      'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, 'LONGITUDE2': lon2, 'PRESSURE': level, 'EOF_NUMBER': int(eofn)}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'PVU', 'description': '{0} hPa PV ensemble mean'.format(level)}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'PVU', 'description': '{0} hPa PV EOF pattern'.format(level)}, 'data': dpvort}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'PV PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr, metname))


    def __temp_eof(self):
        '''
        Function that computes SLP EOF metric, which is calculated by taking the EOF of 
        the ensemble SLP forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean SLP pattern 
        along with the SLP perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('temp_metric_loc'),self.datea_str)):

           print(infull)

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr   = int(conf['definition'].get('forecast_hour'))
              lat1  = float(conf['definition'].get('latitude_min'))
              lat2  = float(conf['definition'].get('latitude_max'))
              lon1  = float(conf['definition'].get('longitude_min'))
              lon2  = float(conf['definition'].get('longitude_max'))
              tmin  = float(conf['definition'].get('temperature_min','0'))
              tmax  = float(conf['definition'].get('temperature_max','400'))
              eofn  = int(conf['definition'].get('eof_number',1))
              level = float(conf['definition'].get('pressure', 850))
              metname = conf['definition'].get('metric_name','t{0}hPa'.format(int(level)))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute temperature EOF'.format(infull))
              return None

           if eval(self.config['model'].get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

#           inpath, infile = infull.rsplit('/', 1)
#           infile1, metname = infile.split('_', 1)
           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                    'description': 'Temperature', 'units': 'K', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

           #  Read the ensemble SLP fields, compute the mean
           for n in range(g1.nens):
              ensmat[n,:,:] = np.squeeze(g1.read_grib_field('temperature', n, vDict))

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config['model'].get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              nlat = len(ensmat[0,:,0])
              nlon = len(ensmat[0,0,:])
              ngrid = -1
              ensarr = xr.DataArray(name='ensemble_data', data=np.zeros([g1.nens, nlat*nlon]), \
                                    dims=['time', 'state'])

              for i in range(nlon):
                 for j in range(nlat):
                    if e_mean[j,i] > tmin and e_mean[j,i] < tmax:
                       ngrid = ngrid + 1
                       ensarr[:,ngrid] = ensmat[:,j,i] * np.sqrt(coslat[j])

           else:

              nlat = len(ensmat[0,:,0])
              nlon = len(ensmat[0,0,:])
              ngrid = -1
              ensarr = xr.DataArray(name='ensemble_data', data=np.zeros([g1.nens, nlat*nlon]), \
                                    dims=['time', 'state'])

              for i in range(nlon):
                 for j in range(nlat):
                    if e_mean[j,i] > tmin and e_mean[j,i] < tmax:
                       ngrid = ngrid + 1
                       ensarr[:,ngrid] = ensmat[:,j,i]


           solver = Eof_xarray(ensarr[:,0:ngrid])
           pcout  = np.squeeze(solver.pcs(npcs=np.maximum(eofn,2), pcscaling=1))
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the SLP pattern associated with a 1 PC perturbation
           dtemp = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              dtemp[:,:] = dtemp[:,:] + ensmat[n,:,:] * pc1[n]

           dtemp[:,:] = dtemp[:,:] / float(g1.nens)

           if np.sum(dtemp) < 0.:
              pc1[:]     = -pc1[:]
              dtemp[:,:] = -dtemp[:,:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#9A32CD","#00008B","#3A5FCD","#00BFFF","#B0E2FF","#FFFFFF","#FFEC8B","#FFA500","#FF4500","#B22222","#FF82AB")
 
           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['fcst_diag'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config['model'].get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the temperature EOF pattern in shading
           tfac = np.ceil(np.max(dtemp) / 2.5)
           cntrs = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]) * tfac
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,dtemp,cntrs,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), extend='both')

           #  Plot the ensemble-mean temperature field in contours
           if level == 850:
              mtemp = np.arange(-16, 12, 2)
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,e_mean - 273.15, \
                              mtemp,linewidths=1.5,colors='k',zorder=10,transform=ccrs.PlateCarree())

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=cntrs)
           cbar.set_ticks(cntrs[1:(len(cntrs)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.title("{0} {1} hour temperature, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           #  Create metric directory, save image file with metric
           outdir = '{0}/f{1}_{2}'.format(self.config['locations']['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           fmetatt = {'FORECAST_METRIC_LEVEL': level, 'FORECAST_METRIC_NAME': 'Temperature PC', 'FORECAST_METRIC_SHORT_NAME': 'tempeof', 'FORECAST_HOUR': int(fhr), \
                      'LATITUDE1': lat1, 'LATITUDE2': lat2, 'LONGITUDE1': lon1, 'LONGITUDE2': lon2, 'PRESSURE': level, \
                      'TEMPERATURE_MIN': tmin, 'TEMPERATURE_MAX': tmax, 'EOF_NUMBER': int(eofn)}

           endict = {'fore_met_init': {'dtype': 'float32'}}

           f_met = {'coords': {}, 'attrs': fmetatt, 'dims': {'num_ens': g1.nens}, 'data_vars': {}}
           f_met['coords']['longitude'] = {'dims': ('longitude'), 'attrs': {'units': 'degrees', 'description': 'longitude of grid points'}, 'data': ensmat.longitude.values}
           endict['longitude'] = {'dtype': 'float32'}
           f_met['coords']['latitude']  = {'dims': ('latitude'), 'attrs': {'units': 'degrees', 'description': 'latitude of grid points'}, 'data': ensmat.latitude.values}
           endict['latitude'] = {'dtype': 'float32'}

           f_met['data_vars']['ensemble_mean'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'K', 'description': '{0} hPa temp. ensemble mean'.format(level)}, 'data': e_mean.data}
           endict['ensemble_mean'] = {'dtype': 'float32'}
           f_met['data_vars']['EOF_pattern'] = {'dims': ('latitude', 'longitude'), 'attrs': {'units': 'K', 'description': '{0} hPa temp. EOF pattern'.format(level)}, 'data': dtemp}
           endict['EOF_pattern'] = {'dtype': 'float32'}
           f_met['data_vars']['fore_met_init'] = {'dims': ('num_ens',), 'attrs': {'units': '', 'description': 'temperature PC'}, 'data': pc1.data}

           xr.Dataset.from_dict(f_met).to_netcdf("{0}/{1}_f{2}_{3}.nc".format(self.config['locations']['work_dir'],str(self.datea_str),'%0.3i' % fhr,metname), encoding=endict)

           self.metlist.append('f{0}_{1}'.format('%0.3i' % fhr, metname))


if __name__ == "__main__":
    src1 = "/Users/parthpatwari/RA_Atmospheric_Science/Old_Code/atcf_data"
    grib_src = "/Users/parthpatwari/RA_Atmospheric_Science/GRIB_files"
    atcf = dpp.Readatcfdata(src1)
    atcf_data = atcf.atcf_files
    no_files = atcf.no_atcf_files
    # g1 = dpp.ReadGribFiles(grib_src, '2019082900', 180)
    ct = ComputeForecastMetrics("ECMWF", '2019082900', atcf.atcf_files, atcf.atcf_array, grib_src)
