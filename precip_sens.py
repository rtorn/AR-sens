import os, sys
import logging
import json
import netCDF4 as nc
import xarray as xr
import numpy as np
import datetime as dt
import matplotlib
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
import cartopy.crs as ccrs

from SensPlotRoutines import plotVecSens, plotScalarSens, computeSens, writeSensFile, set_projection, background_map, addDrop

def ComputeSensitivity(datea, fhr, metname, config):
   '''
   This class is the workhorse of the code because it computes the sensitivity of a given forecast
   metric, which is computed for each ensemble member earlier in the code, to a set of forecast 
   fields at a given forecast hour.  These forecast fields were also computed and placed in separate
   netCDF files in an earlier routine.  The result of this routine is a set of sensitivity graphics
   that can be used for various purposes, and if desired netCDF files that can be ingested into
   alternative software packages for flight planning.  Note that this routine is 
   called several times within the main code; therefore, it could be parallelized.

   Attributes:
        datea   (string):  initialization date of the forecast (yyyymmddhh format)
        fhr        (int):  forecast hour
        metname (string):  name of forecast metric to compute sensitivity for
        config   (dict.):  dictionary that contains configuration options (read from file)
   '''

   plotDict = {}

   for key in config:
      plotDict[key] = config[key]

   for key in config['sens']:
      plotDict[key] = config['sens'][key]

   fhrt = '%0.3i' % fhr

   logging.warning('Sensitivity of {0} to F{1}'.format(metname,fhrt))

   plotDict['plotTitle']    = '{0} F{1}'.format(datea,fhrt)
   plotDict['fileTitle']    = 'AR Recon ECMWF Sensitivity'
   plotDict['initDate']     = '{0}-{1}-{2} {3}:00:00'.format(datea[0:4],datea[4:6],datea[6:8],datea[8:10])
   plotDict['left_labels']  = 'True'
   plotDict['right_labels'] = 'None'      

   if 'ring_center_lat' in config['sens'] and 'ring_center_lon' in config['sens']:
      plotDict['ring_center_lat'] = float(config['sens']['ring_center_lat'])
      plotDict['ring_center_lon'] = float(config['sens']['ring_center_lon'])

   #  Obtain the metric information (here read from file)
   try:
      mfile = nc.Dataset('{0}/{1}_{2}.nc'.format(config['work_dir'],datea,metname))
   except IOError:
      logging.error('{0}/{1}_{2}.nc does not exist'.format(config['work_dir'],datea,metname))
      return

   metric = mfile.variables['fore_met_init'][:]
   nens   = len(metric)
   metric = metric[:] - np.mean(metric, axis=0)

   cmaxmet = np.std(metric) * 0.6
   if cmaxmet >= 1.0:
      plotDict['sensmax'] = np.ceil(cmaxmet)
   else:
      plotDict['sensmax'] = 0.1 * np.ceil(10*cmaxmet)

   init   = dt.datetime.strptime(datea, '%Y%m%d%H')
   datef   = init + dt.timedelta(hours=fhr)
   datef_s = datef.strftime("%Y%m%d%H")
   if 'dropsonde_file' in plotDict:
      plotDict['dropsonde_file'] = plotDict['dropsonde_file'].format(datef_s)

   if 'output_list' in config['sens']:
      flist = [e.strip() for e in config['sens']['output_list'].split(',')]
   else:
      flist = []

   if 'flip_lon' in config:
      plotDict['flip_lon'] = config['flip_lon']
 
   #  Read major axis direction if appropriate  
   if hasattr(mfile.variables['fore_met_init'],'units'):
      plotDict['metricUnits'] = mfile.variables['fore_met_init'].units


   #  Read IVT, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_ivt_ens.nc'.format(config['work_dir'],datea,fhrt)
   if os.path.isfile(ensfile):

      efile = nc.Dataset(ensfile)
      lat   = efile.variables['latitude'][:]
      lon   = efile.variables['longitude'][:]
      ens   = np.squeeze(efile.variables['ensemble_data'][:])
      emea  = np.mean(ens, axis=0)
      emea.units = efile.variables['ensemble_data'].units
      evar = np.var(ens, axis=0)

      plotDict['projinfo'] = set_projection(plotDict.get('projection', 'PlateCarree'), \
                                            float(plotDict.get('min_lon', np.amin(lon))), \
                                            float(plotDict.get('max_lon', np.amax(lon))), plotDict)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/ivt'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if eval(config['sens'].get('output_sens', 'False')) and ('ivt' in flist):
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_ivt_sens.nc'.format(config['figure_dir'],metname,datea,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([200., 400., 600., 800., 1000., 1300., 1600.])
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ivt_sens.png'.format(outdir,datea,fhrt), plotDict)

      mivt = emea[:,:]
      sivt = sens[:,:]


   if 'wind_levels' in config['sens']: 
      plist = json.loads(config['sens'].get('wind_levels'))
   else:
      plist = [1000, 925, 850, 700, 500, 300, 250, 200]

   for pres in plist:

      uensfile = '{0}/{1}_f{2}_uwnd{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      vensfile = '{0}/{1}_f{2}_vwnd{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(uensfile) and os.path.isfile(vensfile):

         efile = nc.Dataset(uensfile)
         lat  = efile.variables['latitude'][:]
         lon  = efile.variables['longitude'][:]
         uens = np.squeeze(efile.variables['ensemble_data'][:])
         umea = np.mean(uens, axis=0)
         uvar = np.var(uens, axis=0)

         efile = nc.Dataset(vensfile)
         vens = np.squeeze(efile.variables['ensemble_data'][:])
         vmea = np.mean(vens, axis=0)
         vvar = np.var(vens, axis=0)

         wmag = np.zeros(umea.shape)
         ivec = np.zeros(umea.shape)
         jvec = np.zeros(umea.shape)
         wmag[:,:] = np.sqrt(umea[:,:]**2+vmea[:,:]**2)
         ivec[:,:] = umea[:,:] / wmag[:,:]
         jvec[:,:] = vmea[:,:] / wmag[:,:]

         ens = np.zeros(uens.shape)

         for n in range(nens):
            ens[n,:,:] = uens[n,:,:] * ivec[:,:] + vens[n,:,:] * jvec[:,:]

         emea = np.mean(ens, axis=0)
         evar = np.var(ens, axis=0)
         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/a{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if plotDict.get('output_sens', 'False')=='True':
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_a{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict) 
         
         plotVecSens(lat, lon, sens, umea, vmea, sigv, '{0}/{1}_f{2}_a{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


         for n in range(nens):
            ens[n,:,:] = uens[n,:,:] * jvec[:,:] - vens[n,:,:] * ivec[:,:]

         emea = np.mean(ens, axis=0)
         evar = np.var(ens, axis=0)
         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/x{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if plotDict.get('output_sens', 'False')=='True':
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_x{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         plotVecSens(lat, lon, sens, umea, vmea, sigv, '{0}/{1}_f{2}_x{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


         sens, sigv = computeSens(uens, umea, uvar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(uvar[:,:])

         outdir = '{0}/{1}/sens/u{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if plotDict.get('output_sens', 'False')=='True':
            writeSensFile(lat, lon, fhr, umea, sens, sigv, '{0}/{1}/{2}_f{3}_u{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)


         sens, sigv = computeSens(vens, vmea, vvar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(vvar[:,:])

         outdir = '{0}/{1}/sens/v{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if plotDict.get('output_sens', 'False')=='True':
            writeSensFile(lat, lon, fhr, vmea, sens, sigv, '{0}/{1}/{2}_f{3}_v{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)


   #  Read theta-e, compute sensitivity to that field
   if 'thetae_levels' in config['sens']:
      plist = json.loads(config['sens'].get('thetae_levels'))
   else:
      plist = [1000, 925, 850, 700, 500, 300, 250, 200]

   for pres in plist:

      ensfile = '{0}/{1}_f{2}_e{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(ensfile):

         efile = nc.Dataset(ensfile)
         lat  = efile.variables['latitude'][:]
         lon  = efile.variables['longitude'][:]
         ens  = np.squeeze(efile.variables['ensemble_data'][:])
         emea = np.mean(ens, axis=0)
         emea.units = efile.variables['ensemble_data'].units
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/e{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if eval(config['sens'].get('output_sens', 'False')) and 'e{0}hPa'.format(pres) in flist:
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_e{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         plotDict['meanCntrs'] = np.array(range(270,390,3))
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_e{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


   plist = [1000, 925, 850, 700, 500, 300, 250, 200]
   for pres in plist:

      ensfile = '{0}/{1}_f{2}_qvap{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(ensfile):

         efile = nc.Dataset(ensfile)
         lat   = efile.variables['latitude'][:]
         lon   = efile.variables['longitude'][:]
         ens   = np.squeeze(efile.variables['ensemble_data'][:])
         emea  = np.mean(ens, axis=0)
         emea.units = efile.variables['ensemble_data'].units
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/qvap{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if eval(config['sens'].get('output_sens', 'False')) and 'qvap{0}hPa'.format(pres) in flist:
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_qvap{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         plotDict['meanCntrs'] = np.array([0.25, 0.50, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_qvap{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


   #  Read PV on various pressure levels, compute sensitivity to that field
   pvd = {'pv200hPa': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], \
          'pv250hPa': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], \
          'pv300hPa': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], \
          'pv500hPa': [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0], \
          'pv700hPa': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4], \
          'pv850hPa': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4]}

   if 'pv_levels' in config['sens']:
      plist = json.loads(config['sens'].get('pv_levels'))
   else:
      plist = [500, 300, 250, 200]

   for pres in plist:

      ensfile = '{0}/{1}_f{2}_pv{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(ensfile): 

         ds = xr.open_dataset(ensfile)
         ens = ds.ensemble_data.squeeze()
         lat = ens.latitude.values
         lon = ens.longitude.values
         emea  = np.mean(ens, axis=0)
         emea.attrs['units'] = ds.ensemble_data.attrs['units']
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/pv{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if eval(config['sens'].get('output_sens', 'False')) and 'pv{0}hPa'.format(pres) in flist:
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_pv{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         if pres >= 500.:
            plotDict['clabel_fmt'] = "%2.1f" 
         plotDict['meanCntrs'] = np.array(pvd['pv{0}hPa'.format(pres)])
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_pv{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)
         if 'clabel_fmt' in plotDict:
            del plotDict['clabel_fmt']          
  

   if eval(config['sens'].get('plot_summary','True')):

      pres = [500, 300, 250, 200]
      nlev = len(pres)

      senslev = []

      for k in range(nlev):

         sensfile = '{0}/{1}/{2}_f{3}_pv{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres[k])
         if os.path.isfile(sensfile):

            efile = nc.Dataset(sensfile)
            senslev.append(np.squeeze(efile.variables['sensitivity'][:]))
            if pres[k] == 250:
               mpv = np.squeeze(efile.variables['ensemble_mean'][:])

         else:

            ds = xr.open_dataset('{0}/{1}_f{2}_pv{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres[k]))
            ens = ds.ensemble_data.values
            emea  = np.mean(ens, axis=0)
            evar = np.var(ens, axis=0)

            if pres[k] == 250:
               mpv = emea[:,:]

            sens, sigv = computeSens(ens, emea, evar, metric)
            sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
            senslev.append(sens)

      pvsens = np.zeros(senslev[0].shape)

      for k in range(nlev-1):

         pvsens[:,:] = pvsens[:,:] + 0.5 * (senslev[k][:,:]+senslev[k+1][:,:]) * abs(pres[k+1]-pres[k])

      pvsens[:,:] = pvsens[:,:] / abs(pres[-1]-pres[0])


      pres = [1000, 925, 850, 700]
      nlev = len(pres)

      senslev = []

      for k in range(nlev):

         sensfile = '{0}/{1}/{2}_f{3}_e{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres[k])
         if os.path.isfile(sensfile):

            efile = nc.Dataset(sensfile)
            senslev.append(np.squeeze(efile.variables['sensitivity'][:]))

         else:

#            ds = xr.open_dataset('{0}/{1}_f{2}_e{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres[k]))
#            ens = ds.ensemble_data.values
            efile = nc.Dataset('{0}/{1}_f{2}_e{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres[k]))
            ens   = np.squeeze(efile.variables['ensemble_data'][:])
            emea  = np.mean(ens, axis=0)
            evar = np.var(ens, axis=0)

            sens, sigv = computeSens(ens, emea, evar, metric)
            sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])
            senslev.append(sens)

      esens = np.zeros(senslev[0].shape)

      for k in range(nlev-1):

         esens[:,:] = esens[:,:] + 0.5 * (senslev[k][:,:]+senslev[k+1][:,:]) * abs(pres[k+1]-pres[k])

      esens[:,:] = esens[:,:] / abs(pres[-1]-pres[0])


      outdir = '{0}/{1}/sens/summ'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      plotSummarySens(lat, lon, mivt, mpv, sivt, esens, pvsens, '{0}/{1}_f{2}_summ_sens.png'.format(outdir,datea,fhrt), plotDict)


   #  Read 300 hPa divergence, compute sensitivity to that field, if the file exists
   ensfile = '{0}/{1}_f{2}_div300hPa_ens.nc'.format(config['work_dir'],datea,fhrt)
   if os.path.isfile(ensfile):

      efile = nc.Dataset(ensfile)
      lat   = efile.variables['latitude'][:]
      lon   = efile.variables['longitude'][:]
      ens   = np.squeeze(efile.variables['ensemble_data'][:])
      emea  = np.mean(ens, axis=0)
      emea.units = efile.variables['ensemble_data'].units
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/div300hPa'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

#      if plotDict.get('output_sens', 'False')=='True':
#         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_div300hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
      plotDict['clabel_fmt'] = "%2.1f"
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_div300hPa_sens.png'.format(outdir,datea,fhrt), plotDict)
      del plotDict['clabel_fmt']

 
   hghtd = {'h500hPa': [4920, 4980, 5040, 5100, 5160, 5220, 5280, 5340, 5400, 5460, 5520, 5580, 5640, 5700, 5760, 5820, 5880, 5940], \
            'h700hPa': [2700, 2730, 2760, 2790, 2820, 2850, 2880, 2910, 2940, 2970, 3000, 3030, 3060, 3090, 3120, 3150, 3180, 3210], \
            'h850hPa': [1110, 1140, 1170, 1200, 1230, 1260, 1290, 1320, 1350, 1380, 1410, 1440, 1470, 1500, 1530, 1560, 1590]}

   #  Read height on various pressure levels, compute sensitivity to that field
   for pres in [500, 700, 850]:

      ensfile = '{0}/{1}_f{2}_h{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(ensfile):

         ds = xr.open_dataset(ensfile)
         ens = ds.ensemble_data.squeeze()
         lat = ens.latitude.values
         lon = ens.longitude.values
         emea  = np.mean(ens, axis=0)
         emea.attrs['units'] = ds.ensemble_data.attrs['units']
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/h{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if eval(config['sens'].get('output_sens', 'False')) and 'h{0}hPa'.format(pres) in flist:
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_h{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         plotDict['meanCntrs'] = np.array(hghtd['h{0}hPa'.format(pres)])
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_h{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


   refd = {'ref850hPa': [205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310], \
           'ref500hPa': [126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177], \
           'ref200hPa': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]}      

   #  Read GPS RO refractivity on pressure levels, compute sensitivity to that field
   for pres in [850, 500, 200]:

      ensfile = '{0}/{1}_f{2}_ref{3}hPa_ens.nc'.format(config['work_dir'],datea,fhrt,pres)
      if os.path.isfile(ensfile):

         ds = xr.open_dataset(ensfile)
         ens = ds.ensemble_data.squeeze()
         lat = ens.latitude.values
         lon = ens.longitude.values
         emea  = np.mean(ens, axis=0)
         emea.attrs['units'] = ds.ensemble_data.attrs['units']      
         evar = np.var(ens, axis=0)

         sens, sigv = computeSens(ens, emea, evar, metric)
         sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

         outdir = '{0}/{1}/sens/ref{2}hPa'.format(config['figure_dir'],metname,pres)
         if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

         if eval(config['sens'].get('output_sens', 'False')) and 'ref{0}hPa'.format(pres) in flist:
            writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_ref{4}hPa_sens.nc'.format(config['figure_dir'],metname,datea,fhrt,pres), plotDict)

         plotDict['meanCntrs'] = np.array(refd['ref{0}hPa'.format(pres)])
         plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_ref{3}hPa_sens.png'.format(outdir,datea,fhrt,pres), plotDict)


   ensfile = '{0}/{1}_f{2}_mslp_ens.nc'.format(config['work_dir'],datea,fhrt)
   if os.path.isfile(ensfile):

      ds = xr.open_dataset(ensfile)
      ens = ds.ensemble_data.squeeze()
      lat = ens.latitude.values
      lon = ens.longitude.values
      emea  = np.mean(ens, axis=0)
      emea.attrs['units'] = ds.ensemble_data.attrs['units']
      evar = np.var(ens, axis=0)

      sens, sigv = computeSens(ens, emea, evar, metric)
      sens[:,:] = sens[:,:] * np.sqrt(evar[:,:])

      outdir = '{0}/{1}/sens/mslp'.format(config['figure_dir'],metname)
      if not os.path.isdir(outdir):
         os.makedirs(outdir, exist_ok=True)

      if eval(config['sens'].get('output_sens', 'False')) and 'mslp' in flist:
         writeSensFile(lat, lon, fhr, emea, sens, sigv, '{0}/{1}/{2}_f{3}_mslp_sens.nc'.format(config['figure_dir'],metname,datea,fhrt), plotDict)

      plotDict['meanCntrs'] = np.array([960, 968, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024, 1028, 1032, 1036, 1040, 1044])
      plotScalarSens(lat, lon, sens, emea, sigv, '{0}/{1}_f{2}_mslp_sens.png'.format(outdir,datea,fhrt), plotDict)


def plotSummarySens(lat, lon, ivt, pvort, ivsens, tesens, pvsens, fileout, plotDict):
   '''
   Function that plots the sensitivity of a forecast metric to a scalar field, along
   with the ensemble mean field in contours, and the statistical significance in 
   stippling.  The user has the option to add customized elements to the plot, including
   range rings, locations of rawinsondes/dropsondes, titles, etc.  These are all turned
   on or off using the configuration file.

   Attributes:
       lat      (float):  Vector of latitude values
       lon      (float):  Vector of longitude values
       sens     (float):  2D array of sensitivity field
       fileout (string):  Name of output figure in .png format
       plotDict (dict.):  Dictionary that contains configuration options
   '''

   minLat = float(plotDict.get('min_lat', np.amin(lat)))
   maxLat = float(plotDict.get('max_lat', np.amax(lat)))
   minLon = float(plotDict.get('min_lon', np.amin(lon)))
   maxLon = float(plotDict.get('max_lon', np.amax(lon)))

   colorivt = ("#FFFFFF", "#FFFC00", "#FFE100", "#FFC600", "#FFAA00", "#FF7D00", \
               "#FF4B00", "#FF1900", "#E60015", "#B3003E", "#80007B", "#570088")
   cnt_ivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
   norm = matplotlib.colors.BoundaryNorm(cnt_ivt,len(cnt_ivt))

   #  Create basic figure, including political boundaries and grid lines
   fig = plt.figure(figsize=plotDict.get('figsize',(11,8.5)))

   ax = background_map(plotDict.get('projection', 'PlateCarree'), minLon, maxLon, minLat, maxLat, plotDict)

#   addRawin(plotDict.get("rawinsonde_file","null"), plt, plotDict)

   plti = plt.contourf(lon, lat, ivt, cnt_ivt, cmap=matplotlib.colors.ListedColormap(colorivt), \
                                 alpha=0.5, norm=norm, extend='max', transform=ccrs.PlateCarree())

   pltpc = plt.contour(lon,lat,pvort,[2.0], linewidths=4.0, colors='k',transform=ccrs.PlateCarree())

   pltis = plt.contour(lon,lat,ivsens,[-0.3, 0.3], linewidths=2.0, colors='g',transform=ccrs.PlateCarree()) 
   pltp = plt.contourf(lon,lat,ivsens,[-0.3, 0.3], hatches=['/', None, '/'], colors='none', \
                        extend='both',transform=ccrs.PlateCarree())
   for i, collection in enumerate(pltp.collections):
      collection.set_edgecolor('g')

   pltc1 = plt.contour(lon,lat,pvsens,[-0.3, 0.3], linewidths=2.0, colors='m',transform=ccrs.PlateCarree())
   pltp = plt.contourf(lon,lat,pvsens,[-0.3, 0.3], hatches=['/', None, '/'], colors='none', \
                       extend='both',transform=ccrs.PlateCarree())
   for i, collection in enumerate(pltp.collections):
      collection.set_edgecolor('m')

   pltc2 = plt.contour(lon,lat,tesens,[-0.3, 0.3], linewidths=2.0, colors='b', \
                             zorder=10, transform=ccrs.PlateCarree())
#   pltt = plt.contourf(lon,lat,tesens,[-0.3, 0.3], hatches=['\\', None, '\\'], colors='none', \
#                       extend='both',transform=ccrs.PlateCarree())
#   for i, collection in enumerate(pltt.collections):
#      collection.set_edgecolor('b')

   if 'plotTitle' in plotDict:
      plt.title(plotDict['plotTitle'])

   #  Add range rings to the file if desired
#   if plotDict.get('range_rings', 'False')=='True' and \
#      'ring_center_lat' in plotDict and 'ring_center_lon' in plotDict:
#      addRangeRings(plotDict['ring_center_lat'], plotDict['ring_center_lon'], lat, lon, plt, plotDict)

   addDrop(plotDict.get("dropsonde_file","null"), plt, plotDict)

   #  Add colorbar to the plot
   cbar = plt.colorbar(plti, fraction=0.15, aspect=45., pad=0.04, orientation='horizontal', ticks=cnt_ivt)
   cbar.set_ticks(cnt_ivt[1:(len(cnt_ivt)-1)])

   plt.savefig(fileout,format='png',dpi=120,bbox_inches='tight')
   plt.close(fig)
