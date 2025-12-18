import os, sys
import glob
import datetime as dt
import configparser
import argparse
import numpy as np

def ar_sens_html(init, metlist, paramfile):

  conf = configparser.ConfigParser()
  conf.read(paramfile)

  #  Now update the list of cases in the sidebar
  with open('{0}/listofcases.txt'.format(conf['locations']['figure_dir']), 'r') as f:
    metfull = [line.rstrip('\n') for line in f]

  init_dt = dt.datetime.strptime(init, '%Y%m%d%H')

  #  Loop over all metrics for this initialization time
  for metric in metlist:

    hhh      = metric.split('_')[0][1:4]
    metname  = metric.split('_')[1]
    figbase  = '{0}/{1}/{2}/sens'.format(conf['locations']['figure_dir'],init,metric)
    htmlbase = '{0}/{1}/{2}/sens'.format(conf['locations']['html_base'],init,metric)

    fhrmax = np.min([int(hhh),int(conf['model']['fcst_hour_max']),int(conf['fields'].get('fields_hour_max',conf['model']['fcst_hour_max']))])

    valid_dt  = init_dt + dt.timedelta(hours=int(hhh))
   
    os.chdir(figbase)

    #  create soft links to each figure, so it is possible to loop over images (needs a specific file name)
    for field in os.listdir(figbase):

      field = field.split('/')[0]

      os.makedirs('{0}/{1}/loop'.format(figbase,field), exist_ok=True)
      os.chdir('{0}/{1}/loop'.format(figbase,field))

      for fhr in range(0,fhrmax+int(conf['model']['fcst_hour_int']),int(conf['model']['fcst_hour_int'])):
        filename = '{0}_f{1}_{2}_sens.png'.format(init,'%0.3i' % fhr,field)
        if (os.path.exists('{0}/{1}/{2}'.format(figbase,field,filename))) and (not os.path.exists('{0}/{1}/loop/f{2}.png'.format(figbase,field,fhr))):
          os.symlink('../{0}'.format(filename), '{0}/{1}/loop/f{2}.png'.format(figbase,field,fhr))

    if os.path.exists('{0}/{1}/{2}/sensitivity.php'.format(conf['locations']['figure_dir'],init,metric)):
      os.remove('{0}/{1}/{2}/sensitivity.php'.format(conf['locations']['figure_dir'],init,metric))

    fout = open('{0}/{1}/{2}/sensitivity.php'.format(conf['locations']['figure_dir'],init,metric), 'w')

    fout.write('  <h2><a href=\"{0}/metric.png\">F{1} metric</a> (valid {2})</h2>\n'.format(metric,hhh,valid_dt.strftime('%H00 UTC %d %b')))

    #  Add code that allows for looping
    fout.write('  <br clear=\'left\'><br>\n')
    fout.write('\n')
    fout.write('  <script src=\'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js\' integrity=\'sha512-894YE6QWD5I59HgZOGReFYm4dnWc1Qt5NtvYSaNcOP+u1T9qYdvdihz0PPSiiqn/+/3e7Jo4EaG7TubfWGUrMQ==\' crossorigin=\'anonymous\' referrerpolicy=\'no-referrer\'></script>\n')
    fout.write('  <script src=\'https://cdnjs.cloudflare.com/ajax/libs/detect_swipe/2.1.4/jquery.detect_swipe.min.js\' integrity=\'sha512-GLP5CTXCqKuoesJrzUxbr9dcSRYaGmjjAHilbGoNszbhba8trPfJH7mLRB8i7JHK6bkaeGkLMYbz/N4B9ndMOQ==\' crossorigin=\'anonymous\' referrerpolicy=\'no-referrer\'></script>\n')
    fout.write('  <script src=\'https://www.atmos.albany.edu/student/abrammer/JsImageloop/JsImageLoop.js\'></script>\n')
    fout.write('  <link rel=\'stylesheet\' type=\'text/css\' href=\'https://www.atmos.albany.edu/student/abrammer/JsImageloop/JsImageLoop.css\'>\n')
    fout.write('  <script>\n')
    fout.write('\n')

    fout.write('  useroptions = {};\n')
    fout.write('  useroptions.content = [];\n')
    fout.write('\n')

    #  Create a line in the loop control panel for each field
    add_field(fout, 'ivt', 'IVT', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)
    add_field(fout, 'iwv', 'IWV', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)
    add_field(fout, 'qvap850hPa', '850 hPa water vapor', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)
    add_field(fout, 'e850hPa', '850 hPa Theta-E', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)
    add_field(fout, 'mslp', 'MSLP', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)
    
    for pres in ['850', '700']:
      add_field(fout, 'h{0}hPa'.format(pres), '{0} hPa Height'.format(pres), figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)

    for pres in ['850', '700', '500', '300', '250']:
      add_field(fout, 'pv{0}hPa'.format(pres), '{0} hPa PV'.format(pres), figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)

    for pres in ['200', '500', '850']:
      add_field(fout, 'ref{0}hPa'.format(pres), '{0} hPa Refrac.'.format(pres), figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)

    add_field(fout, 'summ', 'Summary', figbase, htmlbase, int(conf['model']['fcst_hour_int']), fhrmax)

    fout.write('  </script>\n')
    fout.write('\n')
    fout.write('  <div id=\'content\'></div>\n')
    fout.close()

    #  Finally, add metric to the list of metrics if it is not already there
    if not any('{0}.{1}.{2}'.format(init,hhh,metname) in item for item in metfull):
      metfull.append('{0}.{1}.{2}'.format(init,hhh,metname))


  #  Now update the list of cases in the sidebar
  os.remove('{0}/.include/nav_metrics.php'.format(conf['locations']['figure_dir']))
  fout = open('{0}/.include/nav_metrics.php'.format(conf['locations']['figure_dir']), 'w')
  fout.write('<div id=\"left_column\">\n')
  fout.write('  <div id=\"menu\">\n')
  fout.write('    <h2> Forecasts </h2>\n')
  fout.write('    <br>\n')
  fout.write('    <center> Choose an initialization time and metric</center>\n')
  fout.write('\n')
  fout.write('    <form action=\"{0}/{1}\" method=\"post\">\n'.format(conf['locations']['html_base'],conf['locations']['html_file']))
  fout.write('\n')
  fout.write('      <select name="case">\n')

  metfull.sort(key=str.lower)
  for metstr in metfull:
    fout.write('            <option value=\"{0}\" <?= $_POST[\'case\']=="{0}\"?\"selected\":\"\"?>>{1} (F{2} {3})</option>\n'.format( \
                           metstr,metstr.split('.')[0],metstr.split('.')[1],metstr.split('.')[2]))

  fout.write('      </select>\n')
  fout.write('      <br>\n')
  fout.write('      <button name=\"submit\" type=\"submit\" value=1 class=\"button\">Submit</button>\n')
  fout.write('\n')
  fout.write('    </form>\n')
  fout.write('    <br>\n')
  fout.write('  </div>\n')
  fout.write('</div>\n')

  fout.close()

  #  write out new list of cases/metrics file
  os.remove('{0}/listofcases.txt'.format(conf['locations']['figure_dir']))
  with open('{0}/listofcases.txt'.format(conf['locations']['figure_dir']), 'w') as f:
    f.writelines([item + '\n' for item in metfull])


def add_field(fid, vname, vdesc, figout, htmlout, fhrint, fhrmax):

  if not os.path.exists('{0}/{1}'.format(figout,vname)):
    return

  fid.write('  useroptions[\'content\'].push(\n')
  fid.write('    {{   title: \'{0}\',\n'.format(vdesc))
  fid.write('             startingframe: 0,\n')
  fid.write('       label_interval: 1,\n')
  fid.write('       labels : fspan(0,{0},{1}),\n'.format(fhrmax,fhrint))
  fid.write('       minval: 0,\n')
  fid.write('       maxval: {0},\n'.format(fhrmax))
  fid.write('       increment: {0},\n'.format(fhrint))
  fid.write('       prefix : \'{0}/{1}/loop/f\',\n'.format(htmlout,vname))
  fid.write('       extension: \'.png\',\n')
  fid.write('     }); \n')
  fid.write('\n')


if __name__ == '__main__':

  #  Read the initialization time and storm from the command line
  exp_parser = argparse.ArgumentParser()
  exp_parser.add_argument('--init',  action='store', type=str, required=True)
  exp_parser.add_argument('--metfile',  action='store', type=str, required=True)
  exp_parser.add_argument('--param',  action='store', type=str, required=True)
  args = exp_parser.parse_args()

  with open(args.metfile, 'r') as f:
    metlist = [line.rstrip('\n') for line in f]

  ar_sens_html(args.init, metlist, args.param)
