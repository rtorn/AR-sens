import os, glob
import sys
import argparse
import importlib
import json
import shutil
import tarfile
import numpy as np
import datetime as dt
import configparser
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

sys.path.append('../esens-util')
import fcst_metrics_ar as fmet
from compute_precip_fields import ComputeFields
import precip_sens as sens
from SensPlotRoutines import background_map

#  Routine to read configuration file
def read_config(datea, filename):
    '''
    This function reads a configuration file, and puts all of the appropriate variables into 
    a nested dictionary that can be passed around the appropriate scripts.  The result is the 
    configuration dictionary.

    Attributes:
        datea  (string):  The initialization time of the forecast (yyyymmddhh)
        filename (dict):  The configuration file with all of the parameters
    '''

    confin = configparser.ConfigParser()
    confin.read(filename)

    config = {}
    config['vitals_plot'] = confin['vitals_plot']
    config['metric']      = confin['metric']
    config['fields']      = confin['fields']
    config['sens']        = confin['sens']
#    config['display']     = confin['display']
    config.update(confin['model'])
    config.update(confin['locations'])

    #  Modify work and output directory for specific case/time
    config['work_dir']   = '{0}/{1}'.format(config['work_dir'],datea)
    config['output_dir'] = '{0}/{1}'.format(config['output_dir'],datea)
    config['figure_dir'] = '{0}/{1}'.format(config['figure_dir'],datea)

    #  Create appropriate directories
    if not os.path.isdir(config['work_dir']):
      try:
        os.makedirs(config['work_dir'])
      except OSError as e:
        raise e

    if (eval(config.get('archive_metric','False')) or eval(config.get('archive_metric','False')) ) and \
               (not os.path.isdir(config['output_dir'])):
      try:
        os.makedirs(config['output_dir'])
      except OSError as e:
        raise e

    if not os.path.isdir(config['figure_dir']):
      try:
        os.makedirs(config['figure_dir'])
      except OSError as e:
        raise e

    return(config)


def main():
    '''
    This is the main routine that calls all of the steps needed to compute ensemble-based
    sensitivity for AR Recon.  The script can be called from the command line, where the
    user inputs the forecast initialization date, and storm name.  The user can also add the
    path to the parameter file.

    Important:  within the parameter file, the user needs to set the variable io_module, which
    contains information for how to read and use grib and ATCF data from a specific source and
    model.  The module specified in this variable will be used to get all input data.

    From command line:

    python run_AR_sens.py -init yyyymmddhh --param paramfile

      where:

        -init is the initialization date in yyyymmddhh format
        -param is the parameter file path (optional, otherwise goes to default values in precip.parm)
    '''

    #  Read the initialization time and storm from the command line
    exp_parser = argparse.ArgumentParser()
    exp_parser.add_argument('--init',  action='store', type=str, required=True)
    exp_parser.add_argument('--param', action='store', type=str)

    args = exp_parser.parse_args()

    datea = args.init

    if args.param:
       paramfile = args.param
    else:
       paramfile = 'precip.parm'

    #  Read the configuration file and set up for usage later
    config = read_config(datea, paramfile)

    #  Import the module that contains routines to read ATCF and Grib data specific to the model
    dpp = importlib.import_module(config['io_module'])

    os.chdir(config['work_dir'])

    for handler in logging.root.handlers[:]:
       logging.root.removeHandler(handler)
    logging.basicConfig(filename="{0}/{1}.log".format(config.get("log_dir","."),datea), \
                               filemode='w', format='%(message)s')
    logging.warning("STARTING SENSITIVITIES for {0}".format(str(datea)))

    #  Copy grib data to the work directory
    dpp.stage_grib_files(datea, config)


    #  Plot the precipitation forecast
    fhr1 = json.loads(config['vitals_plot'].get('precip_hour_1'))
    fhr2 = json.loads(config['vitals_plot'].get('precip_hour_2'))

    for h in range(len(fhr1)):
       precipitation_ens_maps(datea, int(fhr1[h]), int(fhr2[h]), config)


    #  Compute precipitation-related forecast metrics
    met = fmet.ComputeForecastMetrics(datea, config)
    metlist = met.get_metlist()


    #  Compute forecast fields at each desired time to use in sensitivity calculation
    fmaxfld = int(config['fields'].get('fields_hour_max',config['fcst_hour_max']))
    for fhr in range(0,fmaxfld+int(config['fcst_hour_int']),int(config['fcst_hour_int'])):

       ComputeFields(datea, fhr, config)


    #  Compute sensitivity of each metric to forecast fields at earlier times, as specified by the user
    for i in range(len(metlist)):

       #  Limit loop over time to forecast metric lead time (i.e., for a 72 h forecast, do not compute 
       #  the sensitivity to fields beyond 72 h
       a = metlist[i].split('_')
       fhrstr = a[0]
       fhrmax = int(np.min([float(fhrstr[1:4]),float(config['fcst_hour_max']),float(fmaxfld)]))

       for fhr in range(0,fhrmax+int(config['fcst_hour_int']),int(config['fcst_hour_int'])):

          sens.ComputeSensitivity(datea, fhr, metlist[i], config)

    with open('{0}/metric_list'.format(config['work_dir']), 'w') as f:
       for item in metlist:
          f.write("%s\n" % item) 
    f.close()


    #  Save some of the files, if needed
    if ( config.get('archive_metric','False') == 'True' ):
       print("Add capability")

    if ( config.get('archive_fields','False') == 'True' ):
       os.rename('{0}/\*_ens.nc'.format(config['work_dir']), '{0}/.'.format(config['output_dir']))


    #  Create a tar file of gridded sensitivity files, if needed
    if eval(config['sens'].get('output_sens', 'False')):

       os.chdir(config['figure_dir'])

       for met in metlist:

          tarout = '{0}/{1}/{1}_{2}_esens.tar'.format(config['outgrid_dir'],datea,met)
          tar = tarfile.open(tarout, 'w')
          for f in glob.glob('{0}/*sens.nc'.format(met)):
             tar.add(f)
          tar.close()

          for f in glob.glob('{0}/{1}/*sens.nc'.format(config['figure_dir'],met)):
             os.remove(f)


    #  Clean up work directory, if desired
    os.chdir('{0}/..'.format(config['work_dir']))
    if not eval(config.get('save_work_dir','False')):
       shutil.rmtree(config['work_dir'])

 
def precipitation_ens_maps(datea, fhr1, fhr2, config):
    '''
    Function that plots the ensemble precipitation forecast between two forecast hours.

    Attributes:
        datea (string):  initialization date of the forecast (yyyymmddhh format)
        fhr1     (int):  starting forecast hour of the window
        fhr2     (int):  ending forecast hour of the window
        config (dict.):  dictionary that contains configuration options (read from file)
    '''

    dpp = importlib.import_module(config['io_module'])

    lat1 = float(config['vitals_plot'].get('min_lat_precip','30.'))
    lat2 = float(config['vitals_plot'].get('max_lat_precip','52.'))
    lon1 = float(config['vitals_plot'].get('min_lon_precip','-130.'))
    lon2 = float(config['vitals_plot'].get('max_lon_precip','-108.'))

    fff1 = '%0.3i' % fhr1
    fff2 = '%0.3i' % fhr2
    datea_1   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr1)
    date1_str = datea_1.strftime("%Y%m%d%H")
    datea_2   = dt.datetime.strptime(datea, '%Y%m%d%H') + dt.timedelta(hours=fhr2)
    date2_str = datea_2.strftime("%Y%m%d%H")
    fint      = int(config.get('fcst_hour_int','12'))
    g1        = dpp.ReadGribFiles(datea, fhr1, config)

    #  Read the total precipitation for the beginning of the window
    if g1.has_total_precip:

       g1 = dpp.ReadGribFiles(datea, fhr1, config)

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('precipitation', vDict)

       g2 = dpp.ReadGribFiles(datea, fhr2, config)

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

    else:

       g1 = dpp.ReadGribFiles(datea, fhr1+fint, config)

       vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                'description': 'precipitation', 'units': 'mm', '_FillValue': -9999.}
       vDict = g1.set_var_bounds('precipitation', vDict)

       ensmat = g1.create_ens_array('precipitation', g1.nens, vDict)

       for n in range(g1.nens):
          ensmat[n,:,:] = np.squeeze(g1.read_grib_field('precipitation', n, vDict))

       for fhr in range(fhr1+2*fint, fhr2+fint, fint):

          print('forecast hour',fhr)
          g1 = dpp.ReadGribFiles(datea, fhr, config)

          for n in range(g1.nens):
             ensmat[n,:,:] = ensmat[n,:,:] + np.squeeze(g1.read_grib_field('precipitation', n, vDict))
       
       if hasattr(g1.read_grib_field('precipitation', 0, vDict), 'units'):
          if g1.read_grib_field('precipitation', 0, vDict).units == "m":
             vscale = 1000.
          else:
             vscale = 1.
       else:
          vscale = 1.

    #  Scale all of the rainfall to mm and to a 24 h precipitation
    ensmat[:,:,:] = ensmat[:,:,:] * vscale * 24. / float(fhr2-fhr1)

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
    plotBase['grid_interval'] = config['vitals_plot'].get('grid_interval', 5)
    plotBase['left_labels'] = 'True'
    plotBase['right_labels'] = 'None'
    ax1 = background_map(config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

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
    ax2 = background_map(config.get('projection', 'PlateCarree'), lon1, lon2, lat1, lat2, plotBase)

    #  Plot the standard deviation of the ensemble precipitation
    spcp = [0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 43.]
    norm = matplotlib.colors.BoundaryNorm(spcp,len(spcp))
    pltf2 = plt.contourf(ensmat.longitude.values,ensmat.latitude.values,e_std,spcp,norm=norm,extend='max', \
                         cmap=matplotlib.colors.ListedColormap(colorlist), transform=ccrs.PlateCarree())

    cbar = plt.colorbar(pltf2, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=spcp)
    cbar.set_ticks(spcp[1:(len(spcp)-1)])

    plt.title('Standard Deviation')

    fig.suptitle('F{0}-F{1} Precipitation ({2}-{3})'.format(fff1, fff2, date1_str, date2_str), fontsize=16)

    outdir = '{0}/std/pcp'.format(config['figure_dir'])
    if not os.path.isdir(outdir):
       try:
          os.makedirs(outdir)
       except OSError as e:
          raise e

    plt.savefig('{0}/{1}_f{2}_pcp24h_std.png'.format(outdir,datea,fff2),format='png',dpi=120,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

   main()
