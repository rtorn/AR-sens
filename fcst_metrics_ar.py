import os, sys, glob
import pandas as pd
import numpy as np
import xarray as xr
import json
import numpy as np
import datetime as dt
import logging
import configparser

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
        self.outdir = config['output_dir']

        self.config = config

        self.metlist = []
        self.dpp = importlib.import_module(config['io_module'])

        if self.config['metric'].get('precipitation_mean_metric', 'True') == 'True':
           self.__precipitation_mean()

        #  Compute precipitation EOF metric
        if self.config['metric'].get('precipitation_eof_metric', 'True') == 'True':
           self.__precipitation_eof()

        #  Compute IVT EOF metric
        if self.config['metric'].get('ivt_eof_metric', 'False') == 'True':
           self.__ivt_eof()

        #  Compute IVT EOF metric
        if self.config['metric'].get('ivt_landfall_metric', 'False') == 'True':
           self.__ivt_landfall_eof()

        #  Compute basin EOF metric
        if self.config['metric'].get('basin_metric', 'False') == 'True':
           self.__precip_basin_eof()

        #  Compute SLP EOF metric
        if self.config['metric'].get('slp_eof_metric', 'False') == 'True':
           self.__slp_eof()

        if self.config['metric'].get('hght_eof_metric', 'False') == 'True':
           self.__hght_eof()

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
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT EOF'.format(infull))
              return None

           if eval(self.config.get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           fff = '%0.3i' % fhr

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

           fDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (300, 1000),
                    'description': 'Integrated Water Vapor Transport', 'units': 'hPa', '_FillValue': -9999.}
           fDict = g1.set_var_bounds('temperature', vDict)

           if 'ivt' in g1.var_dict:

              for n in range(g1.nens):
                 ensmat[n,:,:] = g1.read_grib_field('ivt', n, vDict)

           else:

              for n in range(g1.nens):

                 uwnd = g1.read_grib_field('zonal_wind', n, fDict) * units('m / sec')
                 vwnd = g1.read_grib_field('meridional_wind', n, fDict) * units('m / sec')

                 tmpk = np.squeeze(g1.read_grib_field('temperature', n, fDict)) * units('K')
                 pres = (tmpk.isobaricInhPa.values * units.hPa).to(units.Pa)

                 if g1.has_specific_humidity:
                    qvap = np.squeeze(g1.read_grib_field('specific_humidity', n, fDict)) * units('dimensionless')
                 else:
                    relh = np.minimum(np.maximum(g1.read_grib_field('relative_humidity', n, fDict), 0.01), 100.0) * units('percent')
                    qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)

                 #  Integrate water vapor over the pressure levels
                 usum = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
                 vsum = np.abs(np.trapz(vwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
                 ensmat[n,:,:] = np.sqrt(usum[:,:]**2 + vsum[:,:]**2)

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config.get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              wgts = np.sqrt(coslat)[..., np.newaxis]
              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

           else:

              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))

           pcout  = solver.pcs(npcs=eofn, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the IVT pattern associated with a 1 PC perturbation
           divt = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              divt[:,:] = divt[:,:] + ensmat[n,:,:] * pc1[n]

           divt[:,:] = divt[:,:] / float(g1.nens)

           if np.sum(divt) < 0.0:
              divt[:,:] = -divt[:,:]
              pc1[:]    = -pc1[:]

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(8.5,11))

           colorlist = ("#FFFFFF", "#FFFC00", "#FFE100", "#FFC600", "#FFAA00", "#FF7D00", \
                        "#FF4B00", "#FF1900", "#E60015", "#B3003E", "#80007B", "#570088")

           plotBase = self.config.copy()
           plotBase['grid_interval'] = self.config['vitals_plot'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           mivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
           norm = matplotlib.colors.BoundaryNorm(mivt,len(mivt))
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_mean,mivt,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           ivtfac = np.floor(np.log10(np.max(np.abs(divt))))
           cntrs = np.array([-9., -8., -7., -6., -5., -4., -3., -2., -1.5, -1., -0.8, -0.6, 0.6, 0.8, 1., 1.5, 2., 3., 4., 5., 6., 7., 8., 9.]) * (10**ivtfac)
           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,divt,cntrs,linewidths=1.5, \
                                transform=ccrs.PlateCarree(),colors='k',zorder=10)

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mivt)
           cbar.set_ticks(mivt[1:(len(mivt)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           plt.title("{0} {1} hour IVT, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           outdir = '{0}/f{1}_{2}eof'.format(self.config['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_pcpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'IVT PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'ivteof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'IVT PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_pcpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}eof.nc".format(self.config['work_dir'], str(self.datea_str), fff, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}eof'.format(fff,metname))


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
           f = open(self.config['metric'].get('coast_points_file'), 'r')
        except IOError:
           logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(self.config['metric'].get('coast_points_file')))
           return None

        incoast = np.array(f.readlines())

        f.close()

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('ivt_land_metric_loc'),self.datea_str)):

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr1 = int(conf['definition'].get('forecast_hour1',0))
              fhr2 = int(conf['definition'].get('forecast_hour2',120))
              metname = conf['definition'].get('metric_name','ivtland')
              eofn = int(conf['definition'].get('eof_number',1))
              latcoa1 = float(conf['definition'].get('latitude_min', 25.0))
              latcoa2 = float(conf['definition'].get('latitude_max', 55.0))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(infull))
              return None

           latcoa = []
           loncoa = []
           latlist = []
           lonlist = []

           for i in range(incoast.shape[0]):
              latcoa.append(float(incoast[i][0:4]))
              loncoa.append(float(incoast[i][5:11]))
              if float(incoast[i][0:4]) >= latcoa1 and float(incoast[i][0:4]) <= latcoa2:
                 latlist.append(float(incoast[i][0:4]))
                 lonlist.append(float(incoast[i][5:11]))

           if eval(self.config.get('flip_lon','False')):
              for i in range(len(loncoa)):
                 loncoa[i] = (loncoa[i] + 360.) % 360.
              for i in range(len(lonlist)):
                 lonlist[i] = (lonlist[i] + 360.) % 360.

           lat1 = np.min(latlist)
           lat2 = np.max(latlist)
           lon1 = np.min(lonlist)
           lon2 = np.max(lonlist)

           fff = '%0.3i' % fhr2

           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('temperature', vDict)

           ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

           fDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (300, 1000),
                    'description': 'Integrated Water Vapor Transport', 'units': 'hPa', '_FillValue': -9999.}
           fDict = g1.set_var_bounds('temperature', vDict)

           fhrvec = np.arange(fhr1, fhr2+int(self.config['fcst_hour_int']), int(self.config['fcst_hour_int']))
           ntime  = len(fhrvec)

           ivtarr = xr.DataArray(name='ensemble_data', data=np.zeros([g1.nens, len(latlist), len(fhrvec)]), dims=['ensemble', 'latitude', 'ftime'], \
                                 coords={'ensemble': [i for i in range(g1.nens)], 'ftime': fhrvec, 'latitude': latlist})

           vecloc = len(np.shape(ensmat.latitude)) == 1

           if not vecloc:

              xloc = np.zeros(len(latlist), dtype=int)
              yloc = np.zeros(len(latlist), dtype=int)

              for i in range(len(latlist)):

                 abslat = np.abs(ensmat.latitude-latlist[i])
                 abslon = np.abs(ensmat.longitude-lonlist[i])
                 c = np.maximum(abslon, abslat)

                 ([yloc[i]], [xloc[i]]) = np.where(c == np.min(c))

           for t in range(ntime):

              g1 = self.dpp.ReadGribFiles(self.datea_str, int(fhrvec[t]), self.config)

              if 'ivt' in g1.var_dict:

                 for n in range(g1.nens):
                    ensmat[n,:,:] = g1.read_grib_field('ivt', n, vDict)

              else:

                 for n in range(g1.nens):

                    uwnd = g1.read_grib_field('zonal_wind', n, fDict) * units('m / sec')
                    vwnd = g1.read_grib_field('meridional_wind', n, fDict) * units('m / sec')

                    tmpk = np.squeeze(g1.read_grib_field('temperature', n, fDict)) * units('K')
                    pres = (tmpk.isobaricInhPa.values * units.hPa).to(units.Pa)

                    if g1.has_specific_humidity:
                       qvap = np.squeeze(g1.read_grib_field('specific_humidity', n, fDict)) * units('dimensionless')
                    else:
                       relh = np.minimum(np.maximum(g1.read_grib_field('relative_humidity', n, fDict), 0.01), 100.0) * units('percent')
                       qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)

                    #  Integrate water vapor over the pressure levels
                    usum = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
                    vsum = np.abs(np.trapz(vwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
                    ensmat[n,:,:] = np.sqrt(usum[:,:]**2 + vsum[:,:]**2)

              if vecloc:

                 for i in range(len(latlist)):
                    ivtarr[:,i,t] = ensmat.sel(latitude=slice(latlist[i], latlist[i]), \
                                               longitude=slice(lonlist[i], lonlist[i])).squeeze()

              else:

                 for i in range(len(latlist)):
                    ivtarr[:,i,t] = ensmat.sel(lat=yloc[i], lon=xloc[i]).squeeze().data

           e_mean = np.mean(ivtarr, axis=0)
           for n in range(g1.nens):
              ivtarr[n,:,:] = ivtarr[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
           wgts = np.sqrt(coslat)[..., np.newaxis]

           solver = Eof_xarray(ivtarr.rename({'ensemble': 'time'}))
           pcout  = solver.pcs(npcs=eofn, pcscaling=1)
           pc1 = np.squeeze(pcout[:,eofn-1])
           pc1[:] = pc1[:] / np.std(pc1)

           #  Compute the IVT pattern associated with a 1 PC perturbation
           divt = np.zeros(e_mean.shape)

           for n in range(g1.nens):
              divt[:,:] = divt[:,:] + ivtarr[n,:,:] * pc1[n]

           divt[:,:] = divt[:,:] / float(g1.nens)


           fig = plt.figure(figsize=(10, 6))

           ax0 = fig.add_subplot(121)
           ax0.set_facecolor("gainsboro")

           mivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
           norm = matplotlib.colors.BoundaryNorm(mivt,len(mivt))
           pltf = ax0.contourf(fhrvec,latlist,e_mean,mivt, \
                                cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           ivtfac = np.max(abs(divt))
           if ivtfac < 60:
             cntrs = np.array([-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
           elif ivtfac >= 60 and ivtfac < 200:
             cntrs = np.array([-180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180])
           else:
             cntrs = np.array([-500, -400, -300, -200, -100, 100, 200, 300, 400, 500])

           pltm = ax0.contour(fhrvec,latlist,divt,cntrs,linewidths=1.5, colors='k', zorder=10)

           ax0.set_xlim([np.min(fhrvec), np.max(fhrvec)])
           ax0.set_ylim([np.min(latcoa), np.max(latcoa)])

           #  Add colorbar to the plot
#           cbar = plt.colorbar(pltf, ax=ax0, fraction=0.15, aspect=45., pad=0.02, ticks=mivt)
#           cbar.set_ticks(mivt[1:(len(mivt)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           plt.xticks(np.arange(fhr1, fhr2, step=12.))
           plt.xlabel('Forecast Hour')
           plt.ylabel('Latitude')

           plotBase = self.config.copy()
           plotBase['subplot']       = 'True'
           plotBase['subrows']       = 1
           plotBase['subcols']       = 2
           plotBase['subnumber']     = 2
           plotBase['grid_interval'] = self.config['vitals_plot'].get('grid_interval', 5)
           plotBase['left_labels'] = 'None'
           plotBase['right_labels'] = 'None'
           plotBase['bottom_labels'] = 'None'
           ax1 = background_map('PlateCarree', -133, -111, np.min(latcoa), np.max(latcoa), plotBase)

           ax1.plot(loncoa, latcoa, 'o', color='black', markersize=6, transform=ccrs.PlateCarree())

           fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           plt.suptitle("{0} {1}-{2} hour IVT, {3} of variance".format(str(self.datea_str),fhr1,fhr2,fracvar))

           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.13, orientation='horizontal', cax=fig.add_axes([0.15, 0.01, 0.7, 0.025]))
           cbar.set_ticks(mivt[1:(len(mivt)-1)])

           outdir = '{0}/f{1}_{2}'.format(self.config['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_pcpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'IVT Landfall PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'ivteof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'IVT PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_pcpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}.nc".format(self.config['work_dir'], str(self.datea_str), fff, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}'.format(fff,metname))


    def __precipitation_mean(self):
        '''
        Function that computes the average precipitation over a polygon using the 
        domain defined by the user in a text file.  
        The function also plots a figure showing the ensemble mean and standard deviation of
        precipitation pattern along with the metric area.
        '''

        search_max = 150.

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('precip_metric_loc'),self.datea_str)):

           try:
              f = open(infull, 'r')
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute precip EOF'.format(infull))
              return None

           #  Read the text file that contains information on the precipitation metric
           fhr1 = int(f.readline())
           fhr2 = int(f.readline())
           latc = float(f.readline())
           lonc = float(f.readline())
           maxf = float(f.readline())
#           lat1 = float(f.readline())
#           lon1 = float(f.readline())
#           lat2 = float(f.readline())
#           lon2 = float(f.readline())

           f.close()

           lat1 = float(self.config['metric'].get('min_lat_precip','30.'))
           lat2 = float(self.config['metric'].get('max_lat_precip','52.'))
           lon1 = float(self.config['metric'].get('min_lon_precip','-130.'))
           lon2 = float(self.config['metric'].get('max_lon_precip','-108.'))

           if eval(self.config.get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.
              lonc = (lonc + 360.) % 360.

           inpath, infile = infull.rsplit('/', 1)
           infile1, metname = infile.split('_', 1)
           fff1 = '%0.3i' % fhr1
           fff2 = '%0.3i' % fhr2
           datea_1   = self.datea + dt.timedelta(hours=fhr1)
           date1_str = datea_1.strftime("%Y%m%d%H")
           datea_2   = self.datea + dt.timedelta(hours=fhr2)
           date2_str = datea_2.strftime("%Y%m%d%H")

           #  Read the total precipitation for the two times that make up the window
           g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1, self.config)

           vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                    'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
           vDict = g1.set_var_bounds('precipitation', vDict)

           g2 = self.dpp.ReadGribFiles(self.datea_str, fhr2, self.config)

           ensmat = g2.create_ens_array('precipitation', g2.nens, vDict)

           for n in range(g2.nens):
              ens1 = np.squeeze(g1.read_grib_field('precipitation', n, vDict))
              ens2 = np.squeeze(g2.read_grib_field('precipitation', n, vDict))
              ensmat[n,:,:] = ens2[:,:] - ens1[:,:]

           if hasattr(ens2, 'units'):
              if ens2.units == "m":
                 vscale = 1000.
              else:
                 vscale = 1.
           else:
              vscale = 1.

           #  Scale all of the rainfall to mm and to a 24 h precipitation
           ensmat[:,:,:] = ensmat[:,:,:] * vscale * 24. / float(fhr2-fhr1)

           lonarr, latarr = np.meshgrid(ens2.longitude.values, ens2.latitude.values)
           cdist = great_circle(lonc, latc, lonarr, latarr)
           nlon  = len(ens2.longitude.values)
           nlat  = len(ens2.latitude.values)

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

           #  start from location of max. SD, search for contiguous points above threshold
           for r in range(1,max(nlon,nlat)):

              i1 = max(icen-r,0)
              i2 = min(icen+r,nlon-1)
              j1 = max(jcen-r,0)
              j2 = min(jcen+r,nlat-1)

              nring = 0
              for i in range(i1+1, i2):
                 if ( e_std[j1,i] >= stdmax and fmgrid[j1+1,i] == 1. ):
                    nring = nring + 1
                    fmgrid[j1,i] = 1.
                 if ( e_std[j2,i] >= stdmax and fmgrid[j2-1,i] == 1. ):
                    nring = nring + 1
                    fmgrid[j2,i] = 1.

              for j in range(j1+1, j2):
                 if ( e_std[j,i1] >= stdmax and fmgrid[j,i1+1] == 1. ):
                    nring = nring + 1
                    fmgrid[j,i1] = 1.
                 if ( e_std[j,i2] >= stdmax and fmgrid[j,i2-1] == 1. ):
                    nring = nring + 1
                    fmgrid[j,i2] = 1.

              if ( e_std[j1,i1] >= stdmax and (fmgrid[j1,i1+1] == 1. or fmgrid[j1+1,i1] == 1.)):
                 nring = nring + 1
                 fmgrid[j1,i1] = 1.

              if ( e_std[j1,i2] >= stdmax and (fmgrid[j1,i2-1] == 1. or fmgrid[j1+1,i2] == 1.)):
                 nring = nring + 1
                 fmgrid[j1,i2] = 1.

              if ( e_std[j2,i1] >= stdmax and (fmgrid[j2,i1+1] == 1. or fmgrid[j2-1,i1] == 1.)):
                 nring = nring + 1
                 fmgrid[j2,i1] = 1.

              if ( e_std[j2,i2] >= stdmax and (fmgrid[j2,i2-1] == 1. or fmgrid[j2-1,i2] == 1.)):
                 nring = nring + 1
                 fmgrid[j2,i2] = 1.

           fmout = np.zeros(g1.nens)
           npts  = np.sum(fmgrid)

           #  Average precipitation
           for n in range(g1.nens):

              fmout[n] = np.sum(fmgrid[:,:]*ensmat[n,:,:]) / npts

           #  Create basic figure, including political boundaries and grid lines
           fig = plt.figure(figsize=(11,6.5), constrained_layout=True)

           colorlist = ("#FFFFFF", "#00ECEC", "#01A0F6", "#0000F6", "#00FF00", "#00C800", "#009000", "#FFFF00", \
                        "#E7C000", "#FF9000", "#FF0000", "#D60000", "#C00000", "#FF00FF", "#9955C9")

           ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
           ax1 = self.__background_map(ax1, lat1, lon1, lat2, lon2)         

           #  Plot the ensemble-mean precipitation on the left panel
           mpcp = [0.0, 0.25, 0.50, 1., 1.5, 2., 4., 6., 8., 12., 16., 24., 32., 64., 96., 97.]
           norm = matplotlib.colors.BoundaryNorm(mpcp,len(mpcp))
           pltf1 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_mean,mpcp, \
                                 cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           cbar = plt.colorbar(pltf1, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mpcp)
           cbar.set_ticks(mpcp[1:(len(mpcp)-1):2])

           plt.title('Mean')

           ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
           ax2 = self.__background_map(ax2, lat1, lon1, lat2, lon2) 

           #  Plot the ensemble standard deviation precipitation on the right panel
           spcp = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 43.]
           norm = matplotlib.colors.BoundaryNorm(spcp,len(spcp))
           pltf2 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_std,spcp, \
                                 cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

           pltm = plt.contour(ensmat.longitude.values,ensmat.latitude.values,fmgrid,[0.49, 0.51],linewidths=2.5, colors='w') 

           cbar = plt.colorbar(pltf2, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=spcp)
           cbar.set_ticks(spcp[1:(len(spcp)-1)])

           plt.title('Standard Deviation')

           fig.suptitle('F{0}-F{1} Precipitation ({2}-{3})'.format(fff1, fff2, date1_str, date2_str), fontsize=16)

           outdir = '{0}/f{1}_{2}'.format(self.config['figure_dir'],fff2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir),format='png',dpi=120,bbox_inches='tight')
           plt.close(fig)

           f_metric = {'coords': {},
                       'attrs': {'FORECAST_METRIC_LEVEL': '',
                                 'FORECAST_METRIC_NAME': 'mean precipitation',
                                 'FORECAST_METRIC_SHORT_NAME': 'pcp'},
                       'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                                'attrs': {'units': 'mm',
                                                                         'description': 'precipitation'},
                                                               'data': fmout}}}

           xr.Dataset.from_dict(f_metric).to_netcdf(
               "{0}/{1}_f{2}_{3}.nc".format(self.config['work_dir'], str(self.datea_str), fff2, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}'.format(fff2,metname))


    def __precipitation_eof(self):
        '''
        Function that computes precipitation EOF metric, which is calculated by taking the EOF of 
        the ensemble precipitation forecast over a domain defined by the user in a text file.  
        The resulting forecast metric is the principal component of the
        EOF.  The function also plots a figure showing the ensemble-mean precipitation pattern 
        along with the precipitation perturbation that is consistent with the first EOF. 
        '''

        for infull in glob.glob('{0}/{1}_*'.format(self.config['metric'].get('precip_metric_loc'),self.datea_str)):

           try:
              conf = configparser.ConfigParser()
              conf.read(infull)
              fhr1 = int(conf['definition'].get('forecast_hour1',0))
              fhr2 = int(conf['definition'].get('forecast_hour2',120))
              metname = conf['definition'].get('metric_name','pcp')
              eofn = int(conf['definition'].get('eof_number',1))
              lat1 = float(conf['definition'].get('latitude_min'))
              lat2 = float(conf['definition'].get('latitude_max'))
              lon1 = float(conf['definition'].get('longitude_min'))
              lon2 = float(conf['definition'].get('longitude_max'))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute precip EOF'.format(infull))
              return None

           if eval(self.config.get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           fff2 = '%0.3i' % fhr2
           fint      = int(self.config.get('fcst_hour_int'))
           g1        = self.dpp.ReadGribFiles(self.datea_str, fhr1, self.config)

           #  Read the total precipitation for the beginning of the window
           if g1.has_total_precip:

              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1, self.config)

              vDict = {'latitude': (lat1-0.00001, lat2), 'longitude': (lon1-0.00001, lon2),
                       'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
              vDict = g1.set_var_bounds('precipitation', vDict)

              g2 = self.dpp.ReadGribFiles(self.datea_str, fhr2, self.config)

              ensmat = g2.create_ens_array('precipitation', g2.nens, vDict)

              for n in range(g2.nens):
                 ens1 = np.squeeze(g1.read_grib_field('precipitation', n, vDict))
                 ens2 = np.squeeze(g2.read_grib_field('precipitation', n, vDict))
                 ensmat[n,:,:] = ens2[:,:] - ens1[:,:]

              if hasattr(ens2, 'units'):
                 if ens2.units == "m":
                    vscale = 1000.
                 else:
                    vscale = 1.
              else:
                 vscale = 1.

              g1.close_files()
              g2.close_files()

           else:

              g1 = self.dpp.ReadGribFiles(self.datea_str, fhr1+fint, self.config)

              vDict = {'latitude': (lat1-0.00001, lat2), 'longitude': (lon1-0.00001, lon2),
                       'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
              vDict = g1.set_var_bounds('precipitation', vDict)

              ensmat = g1.create_ens_array('precipitation', g1.nens, vDict)

              for n in range(g1.nens):
                 ensmat[n,:,:] = np.squeeze(g1.read_grib_field('precipitation', n, vDict))

              for fhr in range(fhr1+2*fint, fhr2+fint, fint):

                 g1 = self.dpp.ReadGribFiles(self.datea_str, fhr, self.config)

                 for n in range(g1.nens):
                    ensmat[n,:,:] = ensmat[n,:,:] + np.squeeze(g1.read_grib_field('precipitation', n, vDict))

              if hasattr(g1.read_grib_field('precipitation', 0, vDict), 'units'):
                 if g1.read_grib_field('precipitation', 0, vDict).units == "m":
                    vscale = 1000.
                 else:
                    vscale = 1.
              else:
                 vscale = 1.

              g1.close_files()

           #  Scale all of the rainfall to mm and to a 24 h precipitation
           ensmat[:,:,:] = ensmat[:,:,:] * vscale * 24. / float(fhr2-fhr1)

           e_mean = np.mean(ensmat, axis=0)
           for n in range(g1.nens):
              ensmat[n,:,:] = ensmat[n,:,:] - e_mean[:,:]

           #  Compute the EOF of the precipitation pattern and then the PCs
           if self.config.get('grid_type','LatLon') == 'LatLon':

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

           for n in range(g1.nens):
              dpcp[:,:] = dpcp[:,:] + ensmat[n,:,:] * pc1[n]

           dpcp[:,:] = dpcp[:,:] / float(g1.nens)

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
           plotBase['grid_interval'] = self.config['vitals_plot'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

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

           #  Add colorbar to the plot
           cbar = plt.colorbar(pltf, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=mpcp)
           cbar.set_ticks(mpcp[1:(len(mpcp)-1)])
           cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt="%1.0f")

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
           plt.title("{0} {1}-{2} hour Precipitation, {3} of variance".format(str(self.datea_str),fhr1,fhr2,fracvar))

           outdir = '{0}/f{1}_{2}eof'.format(self.config['figure_dir'],fff2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_pcpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'precipitation PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'pcpeof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'precipitation PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_pcpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}eof.nc".format(self.config['work_dir'], str(self.datea_str), fff2, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}eof'.format(fff2,metname))


    def __precip_basin_eof(self):

        #wget --no-check-certificate https://cw3e.ucsd.edu/Projects/QPF/data/eps_watershed_precip8.nc

        db = pd.read_csv(filepath_or_buffer=self.config['metric'].get('basin_huc_file'), \
                           sep = ',', header=None, skipinitialspace=True, quotechar="\"")
        db.columns = ['ID','Name','Size']

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
              accumulated = eval(conf['definition'].get('accumulation','False'))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute precip EOF'.format(infull))
              return None

           ds = xr.open_dataset('watershed_precip.nc', decode_times=False).rename({'time': 'hour'})

           if hucid_input:

              hucid_list = [e.strip() for e in hucid_input.split(',')]

              basin_list = []
              for hucid in hucid_list:
                  basin_list.append(db[db['ID'] == int(hucid)]['Name'].values)

           else:

              basin_list = [e.strip() for e in basin_input.split(',')]

              hucid_list = []
              for basin in basin_list:
                  hucid_list.append(db[db['Name'] == basin]['ID'].values)

           basinsum = 0.

           ensmat = ((ds.precip.sel(hour=slice(fhr1, fhr2), HUCID=int(hucid_list[0])).squeeze()).transpose()).load()

           for hucid in hucid_list:

              bmask = db['ID'] == int(hucid)
              print('{0} ({1}), {2} Acres'.format(db[bmask]['Name'].values,db[bmask]['ID'].values,np.round(db[bmask]['Size'].values)))

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
                 ylabel = 'Precipitation Rate (mm 6 $\mathregular{h^{-1}}$)'

           
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
           fig = plt.figure(figsize=(10, 6))
           ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#           plt.rcParams.update({'font.size': 15})

           for n in range(ensmat.shape[0]):
              ax.plot(ensmat.hour, ensmat[n,:]+e_mean[:], color='lightgray')

           ax.plot(ensmat.hour, e_mean, color='black', linewidth=3)
           ax.plot(ensmat.hour, e_mean[:]+dpcp[:], '--', color='black', linewidth=3)

           ax.set_xlim((fhr1, fhr2))

           init   = dt.datetime.strptime(self.datea_str, '%Y%m%d%H')
           ticklist = []
           for fhr in np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.):

              datef  = init + dt.timedelta(hours=fhr)
              ticklist.append(datef.strftime("%HZ\n%m/\n%d"))

           ax.set_xticks(np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.))
           ax.set_xticklabels(ticklist, fontsize=15)           

           ax.set_ylabel(ylabel, fontsize=15)
           ax.set_ylim(bottom=0.)

           for label in ax.get_yticklabels():
              label.set_fontsize(15)

           if eofn == 1:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=1)
           else:
              fracvar = '%4.3f' % solver.varianceFraction(neigs=eofn)[-1]
#           plt.title("{0} {1}-{2} hour {3} Precipitation, {4} of variance".format(str(self.datea_str),fhr1,fhr2,\
#                                  basin_name,fracvar))
           plt.title("{0} {1}-{2} hour Precipitation, {3} of variance".format(str(self.datea_str),fhr1,fhr2,\
                                  fracvar), fontsize=15)

           init   = dt.datetime.strptime(self.datea_str, '%Y%m%d%H')
           ticklist = []
           for fhr in np.arange(np.ceil(float(fhr1)/12.)*12., np.floor(float(fhr2)/12.)*12., step=12.):

              datef  = init + dt.timedelta(hours=fhr)
              ticklist.append(datef.strftime("%HZ\n%m/\n%d"))

           print(ticklist)

           outdir = '{0}/f{1}_{2}'.format(self.config['figure_dir'],'%0.3i' % fhr2,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=150, bbox_inches='tight')
           plt.close(fig)

           f_met_basineof_nc = {'coords': {},
                                'attrs': {'FORECAST_METRIC_LEVEL': '',
                                          'FORECAST_METRIC_NAME': 'integrated min. SLP PC',
                                          'FORECAST_METRIC_SHORT_NAME': 'intslp'},
                                'dims': {'num_ens': ensmat.shape[0]},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                              'attrs': {'units': '',
                                                                        'description': 'integrated min. SLP PC'},
                                                               'data': pc1.data}}}

           fff2 = '%0.3i' % fhr2
           xr.Dataset.from_dict(f_met_basineof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}.nc".format(self.config['work_dir'], str(self.datea_str), fff2, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}'.format(fff2,metname))


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
              metname = conf['definition'].get('metric_name','mslp')
              eofn = int(conf['definition'].get('eof_number',1))
              lat1 = float(conf['definition'].get('latitude_min'))
              lat2 = float(conf['definition'].get('latitude_max'))
              lon1 = float(conf['definition'].get('longitude_min'))
              lon2 = float(conf['definition'].get('longitude_max'))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute SLP EOF'.format(infull))
              return None

           if eval(self.config.get('flip_lon','False')):
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
           if self.config.get('grid_type','LatLon') == 'LatLon':

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
           plotBase['grid_interval'] = self.config['vitals_plot'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

           #  Plot the SLP EOF pattern in shading
           slpfac = np.ceil(np.max(np.abs(dslp)) / 5.0)
           cntrs = np.array([-5., -4., -3., -2., -1., 1., 2., 3., 4., 5]) * slpfac
           pltf = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,dslp,cntrs,transform=ccrs.PlateCarree(), \
                                cmap=matplotlib.colors.ListedColormap(colorlist), extend='both')

           #  Plot the ensemble-mean SLP field in contours
           mslp = [976, 980, 984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040]
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
           plt.title("{0} {1} hour Precipitation, {2} of variance".format(str(self.datea_str),fhr,fracvar))

           #  Create a output directory with the metric file
           outdir = '{0}/f{1}_{2}eof'.format(self.config['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_slpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'SLP PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'slpeof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'sea-level pressure PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_slpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}eof.nc".format(self.config['work_dir'], str(self.datea_str), fff, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}eof'.format(fff,metname))


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
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute IVT Landfall EOF'.format(infull))
              return None

           if eval(self.config.get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           inpath, infile = infull.rsplit('/', 1)
           infile1, metname = infile.split('_', 1)
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
           if self.config.get('grid_type','LatLon') == 'LatLon':

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

           outdir = '{0}/f{1}_{2}eof'.format(self.config['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_slpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'Height PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'hghteof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'height PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_slpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}eof.nc".format(self.config['work_dir'], str(self.datea_str), fff, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}eof'.format(fff,metname))


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
              eofn  = int(conf['definition'].get('eof_number',1))
              level = float(conf['definition'].get('pressure', 850))
           except IOError:
              logging.warning('{0} does not exist.  Cannot compute temperature EOF'.format(infull))
              return None

           if eval(self.config.get('flip_lon','False')):
              lon1 = (lon1 + 360.) % 360.
              lon2 = (lon2 + 360.) % 360.

           inpath, infile = infull.rsplit('/', 1)
           infile1, metname = infile.split('_', 1)
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
           if self.config.get('grid_type','LatLon') == 'LatLon':

              coslat = np.cos(np.deg2rad(ensmat.latitude.values)).clip(0., 1.)
              wgts = np.sqrt(coslat)[..., np.newaxis]
              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}), weights=wgts)

           else:

              solver = Eof_xarray(ensmat.rename({'ensemble': 'time'}))

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
           plotBase['grid_interval'] = self.config['vitals_plot'].get('grid_interval', 5)
           plotBase['left_labels'] = 'True'
           plotBase['right_labels'] = 'None'

           ax = background_map(self.config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

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
           outdir = '{0}/f{1}_{2}eof'.format(self.config['figure_dir'],fff,metname)
           if not os.path.isdir(outdir):
              try:
                 os.makedirs(outdir)
              except OSError as e:
                 raise e

           plt.savefig('{0}/metric.png'.format(outdir), format='png', dpi=120, bbox_inches='tight')
           plt.close(fig)

           f_met_tmpeof_nc = {'coords': {},
                              'attrs': {'FORECAST_METRIC_LEVEL': '',
                                        'FORECAST_METRIC_NAME': 'Temperature PC',
                                        'FORECAST_METRIC_SHORT_NAME': 'tempeof'},
                                'dims': {'num_ens': g1.nens},
                                'data_vars': {'fore_met_init': {'dims': ('num_ens',),
                                                               'attrs': {'units': '',
                                                                         'description': 'temperature PC'},
                                                               'data': pc1.data}}}

           xr.Dataset.from_dict(f_met_tmpeof_nc).to_netcdf(
               "{0}/{1}_f{2}_{3}eof.nc".format(self.config['work_dir'], str(self.datea_str), fff, metname), encoding={'fore_met_init': {'dtype': 'float32'}})

           self.metlist.append('f{0}_{1}eof'.format(fff,metname))


if __name__ == "__main__":
    src1 = "/Users/parthpatwari/RA_Atmospheric_Science/Old_Code/atcf_data"
    grib_src = "/Users/parthpatwari/RA_Atmospheric_Science/GRIB_files"
    atcf = dpp.Readatcfdata(src1)
    atcf_data = atcf.atcf_files
    no_files = atcf.no_atcf_files
    # g1 = dpp.ReadGribFiles(grib_src, '2019082900', 180)
    ct = ComputeForecastMetrics("ECMWF", '2019082900', atcf.atcf_files, atcf.atcf_array, grib_src)
