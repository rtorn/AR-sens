import os, sys
import logging
import json
import gc
import importlib
import numpy as np
import datetime as dt
import metpy.constants as mpcon
import metpy.calc as mpcalc
from metpy.units import units

import grid_calc

'''
Class that computes individual 2D ensemble forecast fields for a given forecast hour.  These ensemble fields 
will be used in the next stage of the code to compute the sensitivity.  The result will be a series of netCDF
files, one for each forecast field, that contains all ensemble members.  The forecast fields that are computed
are determined via the configuration options.

Attributes:
    datea (string):  initialization date of the forecast (yyyymmddhh format)
    fhr      (int):  forecast hour
    config (dict.):  dictionary that contains configuration options (read from file)
'''

def ComputeFields(datea, fhr, config):

   # lat_lon info
   lat1 = float(config['fields'].get('min_lat','10.'))
   lat2 = float(config['fields'].get('max_lat','60.'))
   lon1 = float(config['fields'].get('min_lon','-180.'))
   lon2 = float(config['fields'].get('max_lon','-110.'))

   dateadt = dt.datetime.strptime(datea, '%Y%m%d%H')
   fff = str(fhr + 1000)[1:]

   dpp = importlib.import_module(config['model']['io_module'])

   logging.warning("Computing hour {0} ensemble fields".format(fff))

   #  Read grib file information for this forecast hour
   g1 = dpp.ReadGribFiles(datea, fhr, config)

   dencode = {'ensemble_data': {'dtype': 'float32'}, 'latitude': {'dtype': 'float32'},
              'longitude': {'dtype': 'float32'}, 'ensemble': {'dtype': 'int32'}}

   #  Compute the IVT (if desired and file is missing)
   outfile='{0}/{1}_f{2}_ivt_ens.nc'.format(config['locations']['work_dir'],datea,fff)
   if (not os.path.isfile(outfile) and config['fields'].get('calc_ivt','True') == 'True'):

      logging.warning("  Computing IVT")

      vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
               'description': 'Integrated Water Vapor Transport', 'units': 'kg s-1', '_FillValue': -9999.}
      vDict = g1.set_var_bounds('temperature', vDict)

      ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

      if 'ivt' in g1.var_dict:

         for n in range(g1.nens):
            ensmat[n,:,:] = g1.read_grib_field('ivt', n, vDict)      

      else:

         vDict = g1.set_var_bounds('zonal_wind', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (400, 1000), \
                                                  'description': 'Integrated Water Vapor Transport', 'units': 'hPa', '_FillValue': -9999.})
         tDict = g1.set_var_bounds('temperature', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (400, 1000), \
                                                   'description': 'Integrated Water Vapor Transport', 'units': 'hPa', '_FillValue': -9999.})

         for n in range(g1.nens):

            #  Obtain the wind speeds
            uwnd = g1.read_grib_field('zonal_wind', n, vDict)
            vwnd = g1.read_grib_field('meridional_wind', n, vDict)
            uwnd[:,:,:] = np.sqrt(uwnd[:,:,:]**2 + vwnd[:,:,:]**2) * units('m / sec')

            #  Compute the mixing ratio
            pres = (uwnd.isobaricInhPa.values * units.hPa).to(units.Pa)

            if g1.has_specific_humidity:
               qvap = g1.read_grib_field('specific_humidity', n, tDict) * units('dimensionless')
            else:
               tmpk = g1.read_grib_field('temperature', n, tDict) * units('K')
               relh = np.minimum(np.maximum(g1.read_grib_field('relative_humidity', n, tDict), 0.01), 100.0) * units('percent')
               qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[:,None,None], tmpk, relh)
               del tmpk, relh

            #  Integrate water vapor over the pressure levels
            ensmat[n,:,:] = np.abs(np.trapz(uwnd[:,:,:]*qvap[:,:,:], pres, axis=0)) / mpcon.earth_gravity
            del uwnd,vwnd,qvap,pres

      ensmat.to_netcdf(outfile, encoding=dencode)
      del ensmat
      gc.collect()

   elif os.path.isfile(outfile):

      logging.warning("  Obtaining integrated water vapor transport data from {0}".format(outfile))


   #  Read geopotential height from file, if ensemble file is not present
   if config['fields'].get('calc_height','True') == 'True':

      if 'height_levels' in config['fields']:
         height_list = json.loads(config['fields'].get('height_levels'))
      else:
         height_list = (300, 500, 700, 850)

      for level in height_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_h{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa height'.format(levstr))

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level), 
                     'description': '{0} hPa height'.format(levstr), 'units': 'm', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('geopotential_height', vDict)
            ensmat = g1.create_ens_array('geopotential_height', g1.nens, vDict)

            for n in range(g1.nens):
               ensmat[n,:,:] = np.squeeze(g1.read_grib_field('geopotential_height', n, vDict))

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa height data from {1}".format(levstr,outfile))


   #  Read geopotential height from file, if ensemble file is not present
   if eval(config['fields'].get('calc_mslp','True')):

      outfile='{0}/{1}_f{2}_mslp_ens.nc'.format(config['locations']['work_dir'],datea,fff)

      if not os.path.isfile(outfile):

         logging.warning('  Computing MSLP')

         vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2),
                  'description': 'MSLP', 'units': 'hPa', '_FillValue': -9999.}
         vDict = g1.set_var_bounds('sea_level_pressure', vDict)
         ensmat = g1.create_ens_array('sea_level_pressure', g1.nens, vDict)

         for n in range(g1.nens):
            ensmat[n,:,:] = np.squeeze(g1.read_grib_field('sea_level_pressure', n, vDict))*0.01

         ensmat.to_netcdf(outfile, encoding=dencode)
         del ensmat
         gc.collect()

      elif os.path.isfile(outfile):

         logging.warning("  Obtaining MSLP data from {0}".format(outfile))


   #  Compute the water vapor mixing ratio (if desired and file is missing)
   if config['fields'].get('calc_qvapor','True') == 'True':

      if 'qvapor_levels' in config['fields']:
         qvapor_list = json.loads(config['fields'].get('qvapor_levels'))
      else:
         qvapor_list = [850]

      for level in qvapor_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_qvap{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa water vapor mixing ratio'.format(levstr))

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa water vapor mixing ratio'.format(levstr), 'units': 'kg/kg', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('temperature', vDict)

            ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

            for n in range(g1.nens):

               if g1.has_specific_humidity:
                  ensmat[n,:,:] = mpcalc.mixing_ratio_from_specific_humidity(np.squeeze(g1.read_grib_field('specific_humidity', \
                                                                                  n, vDict)) * units('dimensionless')) * 1000.
               else:
                  tmpk = np.squeeze(g1.read_grib_field('temperature', n, vDict))
                  relh = np.squeeze(g1.read_grib_field('relative_humidity', n, vDict))
                  relh[:,:] = np.minimum(np.maximum(relh[:,:], 0.01), 100.0) * units.percent

                  pres = tmpk.isobaricInhPa.values * units.hPa
                  ensmat[n,:,:] = np.squeeze(mpcalc.mixing_ratio_from_relative_humidity(pressure=pres[None,None], \
                                                    temperature=tmpk, relative_humidity=relh)) * 1000.0
                  del tmpk,relh,pres

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa water vapor mixing ratio from {1}".format(levstr,outfile))


   #  Compute the equivalent potential temperature (if desired and file is missing)
   if config['fields'].get('calc_theta-e','False') == 'True':

      if 'thetae_levels' in config['fields']:
         thetae_list = json.loads(config['fields'].get('thetae_levels'))
      else:
         thetae_list = [850]

      for level in thetae_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_e{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa Theta-E'.format(levstr))

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa Equivalent Potential Temperature'.format(levstr), 'units': 'K', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('temperature', vDict)

            ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

            for n in range(g1.nens):

               tmpk = np.squeeze(g1.read_grib_field('temperature', n, vDict)) * units('K')
               pres = tmpk.isobaricInhPa.values * units.hPa

               if g1.has_specific_humidity:
                  qvap = np.squeeze(g1.read_grib_field('specific_humidity', n, vDict)) 
                  qvap[:,:] = np.maximum(qvap[:,:], 1.0e-6) * units('dimensionless')
                  tdew = mpcalc.dewpoint_from_specific_humidity(pres, tmpk, qvap)
               else:
                  relh = np.squeeze(g1.read_grib_field('relative_humidity', n, vDict))
                  relh[:,:] = np.minimum(np.maximum(relh[:,:], 0.01), 100.0) * units.percent
                  qvap = mpcalc.mixing_ratio_from_relative_humidity(pressure=pres[None,None], \
                                                    temperature=tmpk, relative_humidity=relh)
                  tdew = mpcalc.dewpoint_from_relative_humidity(tmpk, relh)

               ensmat[n,:,:] = np.squeeze(mpcalc.equivalent_potential_temperature(pres[None, None], tmpk, tdew))
               del tmpk,tdew,pres

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa Theta-e data from {1}".format(levstr,outfile))


   #  Compute wind-related forecast fields (if desired and file is missing)
   if config['fields'].get('calc_winds','True') == 'True':

      if 'wind_levels' in config['fields']:
         wind_list = json.loads(config['fields'].get('wind_levels'))
      else:
         wind_list = [850]

      for level in wind_list:

         levstr = '%0.3i' % int(level)
         ufile='{0}/{1}_f{2}_uwnd{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)
         vfile='{0}/{1}_f{2}_vwnd{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if (not os.path.isfile(ufile)) or (not os.path.isfile(vfile)):

            logging.warning('  Computing {0} hPa wind information'.format(levstr))

            uDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa zonal wind'.format(levstr), 'units': 'm/s', '_FillValue': -9999.}
            uDict = g1.set_var_bounds('zonal_wind', uDict)

            uensmat = g1.create_ens_array('zonal_wind', g1.nens, uDict)

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa meridional wind'.format(levstr), 'units': 'm/s', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('meridional_wind', vDict)

            vensmat = g1.create_ens_array('meridional_wind', g1.nens, vDict)

            for n in range(g1.nens):

               uwnd = g1.read_grib_field('zonal_wind', n, uDict).squeeze() * units('m/s')
               vwnd = g1.read_grib_field('meridional_wind', n, vDict).squeeze() * units('m/s')

               uensmat[n,:,:] = uwnd[:,:]
               vensmat[n,:,:] = vwnd[:,:]

            uensmat.to_netcdf(ufile, encoding=dencode)
            vensmat.to_netcdf(vfile, encoding=dencode)
            del uensmat,vensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa wind information from file".format(levstr))


   #  Compute divergence forecast fields (if desired and file is missing)
   if config['fields'].get('calc_divergence','False') == 'True':

      if 'divergence_levels' in config['fields']:
         div_list = json.loads(config['fields'].get('divergence_levels'))
      else:
         div_list = [300]

      for level in div_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_div{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa divergence information'.format(levstr))

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa divergence'.format(levstr), 'units': '1/s', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('zonal_wind', vDict)

            ensmat = g1.create_ens_array('zonal_wind', g1.nens, vDict)

            for n in range(g1.nens):

               uwnd = g1.read_grib_field('zonal_wind', n, vDict).squeeze() * units('m/s')
               vwnd = g1.read_grib_field('meridional_wind', n, vDict).squeeze() * units('m/s')

               lat  = ensmat.latitude.values
               lon  = ensmat.longitude.values
               dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, x_dim=-1, y_dim=-2, geod=None)
               div = mpcalc.divergence(uwnd, vwnd, dx=dx, dy=dy)
               ensmat[n,:,:] = grid_calc.calc_circ_llgrid(div, 300., lat, lon, eval(config['fields'].get('global','False')), \
                                                          len(lon), len(lat)) * 1.0e5

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa divergence information from file".format(levstr))


   #  Compute the GPS RO refractivity (if desired and file is missing)
   if config['fields'].get('calc_refractivity','False') == 'True':

      refc1 = 77.6e-6 * units('K/hPa')
      refc2 = 3.73e-1 * units('K^2/hPa')
#      refc1 = 77.6890* units('K/hPa')
#      refc2 = 3.75463e5 * units('K^2/hPa')
      refc3 = 6.3938 * units('K/hPa')
      rdorv = 287.0 / 461.6

      if 'refrac_levels' in config['fields']:
         refrac_list = json.loads(config['fields'].get('refrac_levels'))
      else:
         refrac_list = (200, 250, 500, 700, 850, 925)

      for level in refrac_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_ref{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa refractivity'.format(levstr))

            vDict = {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (level, level),
                     'description': '{0} hPa Refractivity'.format(levstr), 'units': 'K', '_FillValue': -9999.}
            vDict = g1.set_var_bounds('temperature', vDict)

            ensmat = g1.create_ens_array('temperature', g1.nens, vDict)

            for n in range(g1.nens):

               tmpk = np.squeeze(g1.read_grib_field('temperature', n, vDict)) * units('K')
               pres = tmpk.isobaricInhPa.values * units.hPa

               if g1.has_specific_humidity:
                  qvap = np.squeeze(g1.read_grib_field('specific_humidity', n, vDict)) * units('dimensionless')
               else:
                  relh = np.squeeze(g1.read_grib_field('relative_humidity', n, vDict))
                  relh[:,:] = np.minimum(np.maximum(relh[:,:], 0.01), 100.0) * units.percent
                  qvap = mpcalc.mixing_ratio_from_relative_humidity(pres[None,None], tmpk, relh)

               ew   = qvap[:,:] * pres / (rdorv + (1.0-rdorv)*qvap[:,:])
               ensmat[n,:,:] = (refc1 * pres / tmpk[:,:] + refc2 * ew[:,:] / (tmpk[:,:]**2)) * 1.0e6
#               ensmat[n,:,:] = (refc1 * pres / tmpk[:,:] + refc2 * ew[:,:] / (tmpk[:,:]**2)) - \
#                                refc3 * ew[:,:] / tmpk[:,:]

               del tmpk,pres,qvap,ew

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning("  Obtaining {0} hPa Refractivity from {1}".format(levstr,outfile))


   #  Compute the PV on pressure levels if the file does not exist
   if config['fields'].get('calc_pv_pres','True') == 'True':

      if 'pv_levels' in config['fields']:
         pv_list = json.loads(config['fields'].get('pv_levels'))
      else:
         pv_list = (250, 500)

      for level in pv_list:

         levstr = '%0.3i' % int(level)
         outfile='{0}/{1}_f{2}_pv{3}hPa_ens.nc'.format(config['locations']['work_dir'],datea,fff,levstr)

         if not os.path.isfile(outfile):

            logging.warning('  Computing {0} hPa PV'.format(levstr))

            pvec = g1.read_pressure_levels('temperature')
            idx  = np.where(pvec==level)
            lev1 = np.min(pvec[(int(idx[0])-1):(int(idx[0])+2)])
            lev2 = np.max(pvec[(int(idx[0])-1):(int(idx[0])+2)])
            if (level == lev1) or (level==lev2):
               continue

            vDict = g1.set_var_bounds('zonal_wind', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (lev1, lev2), \
                                                     'description': '{0} hPa Potential Vorticity'.format(levstr), 'units': 'PVU', '_FillValue': -9999.})
            tDict = g1.set_var_bounds('temperature', {'latitude': (lat1, lat2), 'longitude': (lon1, lon2), 'isobaricInhPa': (lev1, lev2), \
                                                      'description': '{0} hPa Potential Vorticity'.format(levstr), 'units': 'PVU', '_FillValue': -9999.})

            ensmat = g1.create_ens_array('zonal_wind', g1.nens, vDict)

            for n in range(g1.nens):

               #  Read all the necessary files from file, smooth fields, so sensitivities are useful
               tmpk = g1.read_grib_field('temperature', n, tDict) * units('K')

               lats = ensmat.latitude.values * units('degrees')
               lons = ensmat.longitude.values * units('degrees')
               pres = tmpk.isobaricInhPa.values * units('hPa')

               thta = mpcalc.potential_temperature(pres[:, None, None], tmpk)

               uwnd = g1.read_grib_field('zonal_wind', n, vDict) * units('m/s')
               vwnd = g1.read_grib_field('meridional_wind', n, vDict) * units('m/s')

               dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats, x_dim=-1, y_dim=-2, geod=None)
               dx = np.maximum(dx, 1.0 * units('m'))

               #  Compute PV and place in ensemble array
               if config['model'].get('grid_type','LatLon') == 'LatLon':

                  pvout = np.abs(mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                                dx[None, :, :], dy[None, :, :], lats[None, :, None]))

                  ensmat[n,:,:] = grid_calc.calc_circ_llgrid(np.squeeze(pvout[np.where(pres == level * units('hPa'))[0],:,:]), \
                                                             300., lats, lons, eval(config['fields'].get('global','False')), len(lons), len(lats)) * 1.0e6

               else: 

                  pvout = np.abs(mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                                dx[None, :, :], dy[None, :, :], lats[None, :, :]))

                  ensmat[n,:,:] = grid_calc.calc_circ(np.squeeze(pvout[np.where(pres == level * units('hPa'))[0],:,:]), \
                                                             300000., g1.dx, len(lats[0,:]), len(lats[:,0])) * 1.0e6

               del lats,lons,pres,thta,uwnd,vwnd,dx,dy,pvout 

            ensmat.to_netcdf(outfile, encoding=dencode)
            del ensmat
            gc.collect()

         elif os.path.isfile(outfile):

            logging.warning('  Obtaining {0} hPa PV data from {1}'.format(level,outfile))


   g1.close_files()
   del g1


def read_precip(datea, fhr1, fhr2, conf, vDict):

   dpp = importlib.import_module(conf['model']['io_module'])

   g2 = dpp.ReadGribFiles(datea, fhr2, conf)
   vDict = g2.set_var_bounds('precipitation', vDict)
   ensmat = g2.create_ens_array('precipitation', g2.nens, vDict)

   #  Calculate total precipitation for models that provide total precipitation over model run
   if g2.has_total_precip:

      if fhr1 > 0:
         g1 = dpp.ReadGribFiles(datea, fhr1, conf)
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

      fint = int(conf['metric'].get('fcst_int',6))
      for fhr in range(fhr1+fint, fhr2+fint, fint):
         g1 = dpp.ReadGribFiles(datea, fhr, conf)
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
