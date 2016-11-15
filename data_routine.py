from glob import glob
from numpy import array, genfromtxt
import copy

def add_dropped(n):
    f=open('dropped.txt','a')
    f.write('%d\n'%n)
    f.close()
    
def drop():
    
    allSNIndx = range(len(glob("jla_light_curves/lc*.list")))
    
    d= genfromtxt('dropped.txt')
    
    for el in d:
        allSNIndx.remove(el)
        
    return array(allSNIndx)


def preProcessData(data):
    
    datum = copy.deepcopy(data)
    #datum['Filter'] = ['sdss%s'%a[-1] for a in datum['Filter']]
    if 'VEGA' in datum['MagSys'][0]:
        datum['MagSys'] = ['vega' for a in datum['MagSys']]
    elif 'AB' in datum['MagSys'][0]:
        datum['MagSys'] = ['ab' for a in datum['MagSys']]
    
    #datum['Date'] = datum['Date'] -datum['Date'][0]
    
    datum = datum[datum['Filter'] != 'SWOPE2::u']   #Maybe change this
    #datum = sncosmo.photdata.standardize_data(datum)
    #datum = sncosmo.photdata.normalize_data(datum)
    #datum = sncosmo.fitting.cut_bands(datum,salt)
    return datum


            
                      
