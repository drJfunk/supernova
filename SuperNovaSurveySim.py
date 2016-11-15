import scipy.stats as stats
from astropy.cosmology import FlatwCDM, FlatLambdaCDM
import sncosmo
import numpy as np




class SyntheticSuperNova(object):
    
    
    def __init__(self):
        self.M0     = -19.3
        self.musd   = 0.1
        self.Rc     = 0.1
        self.Rx     = 1.
        self.x1Star = 0.0
        self.cStar  = 0.0
        self._alpha = 0.13
        self._beta  = 2.56
        self.cosmo = FlatwCDM(H0=72.,Om0=.3,w0=-1.)
        self._SetSurveyParams()
        
        self._Generate()
        


        
    def _Generate(self):

        self._GenerateZ()
        self._GenerateM()

        self._GenerateX1()
        self._GenerateC()
        
        self._SetDistMod()
        self._GeneratemB()
    
    
    def _SetSurveyParams(self):
        '''
        virtual function to setup survey params
        '''
        print "In Super Class"
        
    
    def _GenerateM(self):
        
        self.M = stats.norm.rvs(self.M0,self.musd)
    
    def _GenerateX1(self):
        
        
        self.X1sd = self._GetPostiveRVS(self._surveyX1mu,self._surveyX1sd)
        self.x1_true = stats.norm.rvs(self.x1Star,self.Rx)
        self.x1 = stats.norm.rvs(self.x1_true,self.X1sd)
    
    def _GenerateC(self):
        
        
        self.Csd = self._GetPostiveRVS(self._surveyCmu, self._surveyCsd)
        self.c_true = stats.norm.rvs(self.cStar,self.Rc)
        
        
        self.c =  stats.norm.rvs(self.c_true,self.Csd)
        
    def _GeneratemB(self):
        
        self.mb_true = self.dm + self.M - self._alpha*self.x1 + self._beta*self.c
        self.mbsd = self._GetPostiveRVS(self._surveymbmu,self._surveymbsd)
        self.mb = stats.norm.rvs(self.mb_true,self.mbsd)
        
        # I bet there is a brightness cutoff
        # so regen if I violat this
        if self.mb<self._mbLim:
            self._Generate()
        
        
    
    def _GetPostiveRVS(self,mu,sd):
        
        flag = True
        while(flag):
            val = stats.norm.rvs(loc=mu,scale=sd)
            if val>0.:
                flag = False
                
        return val
        
        
    
    def _SetDistMod(self):
        
        
        self.dm = self.cosmo.distmod(self.z).value
        
        
    
    def _GenerateZ(self):
        self.z = self._GetPostiveRVS(self.zmu,self.zsd)
        
    
    
    def GetObsParams(self):
        
        
        
        return np.array([self.z,self.mb,self.mbsd,self.c,self.Csd,self.x1,self.X1sd,self.survey])

    def GetLatentParams(self):
        
        return np.array([self.M,self.mb_true,self.c_true,self.x1_true])
    
    

class HSTSuperNova(SyntheticSuperNova):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.198
        self._surveyCsd  = 0.026
        self._surveymbsd = 0.007
        self._surveyX1mu = .416
        self._surveyCmu  = .055
        self._surveymbmu = .112 
        
        self.zmu = .963
        self.zsd = 0.226
        
        self._mbLim = 24
        
        self.survey = 'hst'
            
    
    
class SDSSSuperNova(SyntheticSuperNova):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = .215
        self._surveyCsd  = 0.012
        self._surveymbsd = .008
        self._surveyX1mu = .375
        self._surveyCmu  = 0.036
        self._surveymbmu = .118
        
        self.zmu = .191
        self.zsd = 0.077
        
        self._mbLim = 16.5
        
        self.survey = 'sdss'
        
class SNLSSuperNova(SyntheticSuperNova):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.194
        self._surveyCsd  = .018
        self._surveymbsd = .01
        self._surveyX1mu = .3
        self._surveyCmu  = .048
        self._surveymbmu = .097
        
        self.zmu = 0.636
        self.zsd = 0.204 
        
        self._mbLim = 19.2

        self.survey = 'snls'


class LowZSuperNova(SyntheticSuperNova):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.089
        self._surveyCsd  = 0.007
        self._surveymbsd = .008
        self._surveyX1mu = .122
        self._surveyCmu  = .028
        self._surveymbmu = .145
        
        self.zmu = .029
        self.zsd = .016
        
        
        self._mbLim = 14.
        self.survey = 'lowz'
        
        

from sklearn.mixture import GMM

class SyntheticSuperNovaMixture(SyntheticSuperNova):


    def __init__(self):
        self.M0A     = -19.1
        self.M0B     = -19.9
        self.musdA   = 0.1
        self.musdB   = 0.1
        self.nA = 0.5
        
        self.Rc     = 0.1
        self.Rx     = 1.
        self.x1Star = 0.0
        self.cStar  = 0.0
        self._alpha = 0.13
        self._beta  = 2.56
        self.cosmo = FlatwCDM(H0=72.,Om0=.3,w0=-1.)
        self._SetSurveyParams()
        
        self._Generate()


    def _nZ(self,z):

        if z<0.05:
            return 1.
        elif z>1.5:
            return 0.

        else:

            return -0.627*z+1.003
        

    def _GenerateM(self):

        gmm = GMM(2)
        gmm.means_ = np.array([[self.M0A], [self.M0B]])
        gmm.covars_ = np.array([[self.musdA], [self.musdB]]) ** 2

        na = .5#self._nZ(self.z)
        nb = 1.-na 
        
        gmm.weights_ = np.array([na, nb])
        self.M = gmm.sample(1)[0][0]
        
        







class HSTSuperNovaMixture(SyntheticSuperNovaMixture):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.198
        self._surveyCsd  = 0.026
        self._surveymbsd = 0.007
        self._surveyX1mu = .416
        self._surveyCmu  = .055
        self._surveymbmu = .112 
        
        self.zmu = .963
        self.zsd = 0.226
        
        self._mbLim = 24
        
        self.survey = 'hst'
            
    
    
class SDSSSuperNovaMixture(SyntheticSuperNovaMixture):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = .215
        self._surveyCsd  = 0.012
        self._surveymbsd = .008
        self._surveyX1mu = .375
        self._surveyCmu  = 0.036
        self._surveymbmu = .118
        
        self.zmu = .191
        self.zsd = 0.077
        
        self._mbLim = 16.5
        
        self.survey = 'sdss'
        
class SNLSSuperNovaMixture(SyntheticSuperNovaMixture):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.194
        self._surveyCsd  = .018
        self._surveymbsd = .01
        self._surveyX1mu = .3
        self._surveyCmu  = .048
        self._surveymbmu = .097
        
        self.zmu = 0.636
        self.zsd = 0.204 
        
        self._mbLim = 19.2

        self.survey = 'snls'


class LowZSuperNovaMixture(SyntheticSuperNovaMixture):
    
    def _SetSurveyParams(self):
        self._surveyX1sd = 0.089
        self._surveyCsd  = 0.007
        self._surveymbsd = .008
        self._surveyX1mu = .122
        self._surveyCmu  = .028
        self._surveymbmu = .145
        
        self.zmu = .029
        self.zsd = .016
        
        
        self._mbLim = 14.
        self.survey = 'lowz'
        

