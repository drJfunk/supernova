import sncosmo
from numpy import array
from glob import glob
import numpy as np
import os

# Change these string to the correct paths
lc_dir = 'jla_light_curves' # JLA lightcurves
snfit_dir = 'snfit_data' # data directory for the SALT fitter

f = open('%s/fitmodel.card'%snfit_dir)
fitmodel_card = {line.split()[0][1:]: line.split()[1] for line in f}
f.close()




#lc_files=lc_files[rndLC_index]

lc_files=array(glob("jla_light_curves/lc*.list"))




filter_names = []
for lc_file in lc_files:
    data = sncosmo.io.read_lc(lc_file, format='salt2')
    filter_names.extend(np.unique(data['Filter']))
    
filter_names = np.unique(sorted(filter_names))


instrument_names = []
for filter_name in filter_names:
    instrument = filter_name.split('::')[0]
    if instrument not in instrument_names:
        instrument_names.append(instrument)


filterwheels = {}
for instrument in instrument_names:
    instrument_dir = fitmodel_card[instrument]
    # Look for FilterWheel file. Unfortunately it is not consistently capitalized...
    # [Insert comment on case-insensitive Mac users here.]
    filterwheel_file = [a for a in os.listdir('%s/%s'%(snfit_dir, instrument_dir))
                        if a.lower() == 'filterwheel'][0]
    f = open('/'.join([snfit_dir, instrument_dir, filterwheel_file]))
    filterwheels[instrument] = {line.split()[0]: line.split()[1:] for line in f}
    f.close()        
        

def get_filter_file(filter_code):
    instrument, filter_name = filter_code.split('::')
    return '/'.join([snfit_dir, fitmodel_card[instrument], filterwheels[instrument][filter_name][-1]])

def register_filter(filter_code, filename, force=False):
    f = open(filename)
    band_data = [line.split() for line in f 
                 if line.split()[0][0] not in ['#', '@']]
    f.close()
    
    wl = np.array([float(a[0]) for a in band_data])
    tr = np.array([float(a[1]) for a in band_data])
    
    bandpass = sncosmo.Bandpass(wl, tr)
    sncosmo.registry.register(bandpass, filter_code, force=force)
    
    
    
for fname in filter_names:
    register_filter(fname, get_filter_file(fname), force=True)
    
    
    
def _get_bandmag(band, magsys, t=0, rest_frame=True, **kwargs):
    """
    Returns mag at max for the model, band and magsys
    Arguments:
    model  -- sncosmo model, e.g. SALT2
    band   -- sncosmo band object or string, e.g. 'bessellb'
    magsys -- magnitude system, e.g. 'ab'
    Keyword arguments:
    t -- time relative to t0 (observer-frame), at which to evaluate
    rest_frame -- default: True, overrides the redshifts
    """
    model = sncosmo.Model(source='salt2')
    if rest_frame:
        kwargs['z'] = 0

    model.set(**kwargs)
    return model.bandmag(band,magsys,kwargs['t0'] + t)

def _get_bandmag_gradient(band, magsys, param, sig, fixed_param, 
                          t=0, rest_frame=True):
    """
    Return gradient of _get_bandmag as function of param
    param, sig must be dictionaries of means and uncertainties
    Best use odicts to make sure that the order of the components is correct
    """
    model = sncosmo.Model(source='salt2')
    out = []
    
    if rest_frame:
        if 'z' in param.keys():
            param['z'] = 0
        if 'z' in fixed_param.keys():
            fixed_param['z'] = 0

    model.set(**fixed_param)
    for key,val in param.items():
        model.set(**param)
        h = sig[key] / 100.
        
        model.set(**{key: val - h})
        m0 = model.bandmag(band, magsys, param['t0'] + t)

        model.set(**{key: val + h})
        m1 = model.bandmag(band, magsys, param['t0'] + t)
        
        out.append((m1 - m0) / (2. * h))

    return np.array(out)
