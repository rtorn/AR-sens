import os, glob
import sys
import argparse
import importlib
import shutil
import tarfile
import numpy as np
import configparser
import logging
from multiprocessing import Pool

sys.path.append('../esens-util')
from fcst_diag import precipitation_ens_maps, basin_ens_maps
import fcst_metrics_ar as fmet
from compute_precip_fields import ComputeFields
from precip_sens import ComputeSensitivity

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
    config['fcst_diag']   = confin['fcst_diag']
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
    if ('precip_hour_1' in config['fcst_diag']) and ('precip_hour_2' in config['fcst_diag']):
       fhr1 = [e.strip() for e in config['fcst_diag'].get('precip_hour_1','').split(',')]
       fhr2 = [e.strip() for e in config['fcst_diag'].get('precip_hour_2','').split(',')]
       if eval(config['fcst_diag'].get('multiprocessor','False')):

          arglist = [(datea, int(fhr1[h]), int(fhr2[h]), config) for h in range(len(fhr1))]
          with Pool() as pool:
             results = pool.map(precipitation_ens_maps_parallel, arglist)

       else:

          for h in range(len(fhr1)):
             precipitation_ens_maps(datea, int(fhr1[h]), int(fhr2[h]), config)

    if ('basin_hour_1' in config['fcst_diag']) and ('basin_hour_2' in config['fcst_diag']):
       fhr1 = [e.strip() for e in config['fcst_diag'].get('basin_hour_1','').split(',')]
       fhr2 = [e.strip() for e in config['fcst_diag'].get('basin_hour_2','').split(',')]
       for h in range(len(fhr1)):
          basin_ens_maps(datea, int(fhr1[h]), int(fhr2[h]), config)


    #  Compute precipitation-related forecast metrics
    met = fmet.ComputeForecastMetrics(datea, config)
    metlist = met.get_metlist()

    #  Exit if there are no metrics
    if len(metlist) < 1:
       logging.error('No metrics have been calculated.  Exiting the program.')
       sys.exit()


    #  Compute forecast fields at each desired time to use in sensitivity calculation
    fmaxfld = int(config['fields'].get('fields_hour_max',config['fcst_hour_max']))
    if eval(config['fields'].get('multiprocessor','False')):

       arglist = [(datea, fhr, config) for fhr in range(0,fmaxfld+int(config['fcst_hour_int']),int(config['fcst_hour_int']))]
       with Pool() as pool:       
          results = pool.map(ComputeFieldsParallel, arglist)

    else:

       for fhr in range(0,fmaxfld+int(config['fcst_hour_int']),int(config['fcst_hour_int'])):
          ComputeFields(datea, fhr, config)


    #  Compute sensitivity of each metric to forecast fields at earlier times
    if eval(config['sens'].get('multiprocessor','False')):    #  Use parallel processing

       fhrarg = []
       metarg = []
       for i in range(len(metlist)):

          #  Limit loop over time to forecast metric lead time
          a = metlist[i].split('_')
          fhrstr = a[0]
          fhrmax = int(np.min([float(fhrstr[1:4]),float(config['fcst_hour_max']),float(fmaxfld)]))

          for fhr in range(0,fhrmax+int(config['fcst_hour_int']),int(config['fcst_hour_int'])):

             fhrarg.append(fhr)
             metarg.append(metlist[i])

       arglist = [(datea, fhrarg[i], metarg[i], config) for i in range(len(fhrarg))]
       with Pool() as pool:
          results = pool.map(ComputeSensitivityParallel, arglist)

    else:  #  Use serial processing

       for i in range(len(metlist)):

          #  Limit loop over time to forecast metric lead time
          a = metlist[i].split('_')
          fhrstr = a[0]
          fhrmax = int(np.min([float(fhrstr[1:4]),float(config['fcst_hour_max']),float(fmaxfld)]))

          for fhr in range(0,fhrmax+int(config['fcst_hour_int']),int(config['fcst_hour_int'])):

             ComputeSensitivity(datea, fhr, metlist[i], config)


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

 
def precipitation_ens_maps_parallel(args):

    datea, fhri, fhrf, config = args
    precipitation_ens_maps(datea, fhri, fhrf, config)


def ComputeFieldsParallel(args):

    datea, fhr, config = args
    ComputeFields(datea, fhr, config)


def ComputeSensitivityParallel(args):

    datea, fhr, metname, config = args
    ComputeSensitivity(datea, fhr, metname, config)


if __name__ == '__main__':

   main()
