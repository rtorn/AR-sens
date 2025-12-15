import os, sys, shutil
import glob
import datetime as dt
import numpy as np
import argparse
import subprocess
import configparser
import urllib.request

from ar_sens_html import ar_sens_html

def init_AR_sens(init, paramfile):

  conf = configparser.ConfigParser()
  conf.read(paramfile)

  init_dt = dt.datetime.strptime(init, '%Y%m%d%H')

  if (not os.path.exists('{0}/pcp/{1}_auto'.format(conf['metric']['precip_metric_loc'],init))) and \
      os.path.exists('{0}/pcp/yyyymmddhh_auto'.format(conf['metric']['precip_metric_loc'])):
    shutil.copy('{0}/pcp/yyyymmddhh_auto'.format(conf['metric']['precip_metric_loc']),'{0}/pcp/{1}_auto'.format(conf['metric']['precip_metric_loc'],init))

  if (not os.path.exists('{0}/rbasin/{1}_auto'.format(conf['metric']['basin_metric_loc'],init))) and \
      os.path.exists('{0}/rbasin/yyyymmddhh_auto'.format(conf['metric']['basin_metric_loc'])):
    shutil.copy('{0}/rbasin/yyyymmddhh_auto'.format(conf['metric']['basin_metric_loc']),'{0}/rbasin/{1}_auto'.format(conf['metric']['basin_metric_loc'],init))

  if (not os.path.exists('{0}/ivtl/{1}_auto'.format(conf['metric']['ivt_land_metric_loc'],init))) and \
      os.path.exists('{0}/ivtl/yyyymmddhh_auto'.format(conf['metric']['ivt_land_metric_loc'])):
    shutil.copy('{0}/ivtl/yyyymmddhh_auto'.format(conf['metric']['ivt_land_metric_loc']),'{0}/ivtl/{1}_auto'.format(conf['metric']['ivt_land_metric_loc'],init))

  conf['locations']['work_dir'] = '{0}/{1}'.format(conf['locations']['work_dir'],init)
  os.makedirs(conf['locations']['work_dir'], exist_ok = True)
  os.chdir(conf['locations']['work_dir'])

  if not os.path.exists('{0}/watershed_precip.nc'.format(conf['locations']['work_dir'])):
    urllib.request.urlretrieve('https://cw3e.ucsd.edu/Projects/QPF/data/watershed_HUC8_eps.nc', 'watershed_precip.nc')

  if (not os.path.exists('{0}/arr_buoys.txt')) and dt.datetime.now().hour > 0:
    fout = open('arr_buoys.txt', 'w')
    for yyyy in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
      urllib.request.urlretrieve('https://cw3e.ucsd.edu/images/CW3E_Obs/DriftingBuoys/ARRecon_{0}_SVP-B_BuoyLocations_latest_SLP.txt'.format(yyyy), 'buoy.txt')
      with open('buoy.txt') as infile:
        fout.write(infile.read())
      os.remove('buoy.txt')

    fout.close()

  if (not os.path.exists('{0}/other_buoys.txt')) and dt.datetime.now().hour > 0:
    urllib.request.urlretrieve('https://cw3e.ucsd.edu/images/CW3E_Obs/DriftingBuoys/BuoyLocations_latest_SLP.txt', 'other_buoys.txt')

  os.chdir('/home11/staff/torn/ens-sens/AR-sens')
  os.system('python run_AR_sens.py --init {0} --param {1}'.format(init,paramfile))

  if conf['model']['io_module'] == 'ecmwf_aws_down' or conf['model']['io_module'] == 'gefs_aws_down': 
    for rmfile in glob.glob('{0}/f*grb2*'.format(conf['locations']['work_dir'])):
      os.remove(rmfile)

  if os.path.exists('{0}/metric_list'.format(conf['locations']['work_dir'])):
    with open('{0}/metric_list'.format(conf['locations']['work_dir']), 'r') as f:
      metlist = [line.rstrip('\n') for line in f]
  else:
    metlist = []

  ar_sens_html(init, metlist, paramfile)

  for mettype in ['pcp1', 'pcp2', 'ivtland1', 'ivtland2', 'ivt1', 'slp1', 'hght1', 'pv1', 'snow1']:

    if os.path.exists('{0}/latest/{1}'.format(conf['locations']['figure_dir'],mettype)):

      for filesuf in ['.png', '.nc', '.tar']:
         for rmfile in glob.glob('{0}/latest/{1}/*{2}'.format(conf['locations']['figure_dir'],mettype,filesuf)):
           os.remove(rmfile)

      if any(mettype in item for item in metlist):

        metname = [item for item in metlist if mettype in item][0]
        
        shutil.copy('{0}/{1}/{2}/metric.png'.format(conf['locations']['figure_dir'],init,metname),'{0}/latest/{1}/.'.format(conf['locations']['figure_dir'],mettype))
        shutil.copy('{0}/{1}_{2}.nc'.format(conf['locations']['work_dir'],init,metname),'{0}/latest/{1}/metric.nc'.format(conf['locations']['figure_dir'],mettype))
        shutil.copy('{0}/{1}/{1}_{2}_esens.tar'.format(conf['locations']['figure_dir'],init,metname),'{0}/latest/{1}/esens.tar'.format(conf['locations']['figure_dir'],mettype))

        for field in ['ivt', 'e850hPa', 'pv500hPa', 'pv250hPa', 'summ']:
          for fhrt in ['048', '072']:
            shutil.copy('{0}/{1}/{2}/sens/{3}/{1}_f{4}_{3}_sens.png'.format(conf['locations']['figure_dir'],init,metname,field,fhrt), \
                        '{0}/latest/{1}/f{2}_{3}_sens.png'.format(conf['locations']['figure_dir'],mettype,fhrt,field))




if __name__ == '__main__':

  #  Read the initialization time and storm from the command line
  exp_parser = argparse.ArgumentParser()
  exp_parser.add_argument('--init',  action='store', type=str, default=dt.datetime.now().strftime("%Y%m%d00"))
  exp_parser.add_argument('--param',  action='store', type=str, required=True)
  args = exp_parser.parse_args()

  init_AR_sens(args.init, args.param)
