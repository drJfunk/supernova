__author__="J. Michael Burgess"

import numpy as np
import matplotlib.pyplot as plt


import theano
import theano.tensor as T
import pymc3 as pm
from cosmo import distmod, distmodW

class SNBayesModel(object):

    def __init__(self,sn_table):



        self._h0 = 67.3 # Planck

        self._zcmb=theano.shared(sn_table['zcmb'])

        self._color=theano.shared(sn_table['color'])

        self._dcolor=theano.shared(sn_table['dcolor'])
        
        self._x1=theano.shared(sn_table['x1'])
        
        self._dx1=theano.shared(sn_table['dx1'])
        
        self._mbObs=theano.shared(sn_table['mb'])
        
        self._dmbObs=theano.shared(sn_table['dmb'])

        self._survey= sn_table['set']
        
        try:

            self._log_host_mass = theano.shared(sn_table['3rdvar'])
            self._dlog_host_mass = theano.shared(sn_table['d3rdvar'])

        except:

            raise RuntimeWarning("No host galacy mass in data")
            
            self._log_host_mass = None
            self._dlog_host_mass = None

        

        self._n_SN = len(sn_table)


        self._data_set = sn_table

        self._create_subsurvey_set()

        
        self._model = pm.Model()
        

        self._model_setup()


        self._advi_complete = False

        self._trace = None
         
    def _model_setup(self):


        print "Super Class"

        pass



    def _create_subsurvey_set(self):



        zcmb_survey=[]

        color_survey=[]

        dcolor_survey=[]
        
        x1_survey=[]
        
        dx1_survey=[]
        
        mbObs_survey=[]
        
        dmbObs_survey=[]

        n_SN_survey=[]


        self._category = np.array([0,1,2,3])
        
        for set_number in [1,2,3,4]:


            this_survey = self._data_set['set'] == set_number

            zcmb_survey.append(np.array(self._data_set['zcmb'][this_survey]))
            
            color_survey.append(np.array(self._data_set['color'][this_survey]))

            dcolor_survey.append(np.array(self._data_set['dcolor'][this_survey]))

            x1_survey.append(np.array(self._data_set['x1'][this_survey]))

            dx1_survey.append(np.array(self._data_set['dx1'][this_survey]))

            mbObs_survey.append(np.array(self._data_set['mb'][this_survey]))

            dmbObs_survey.append(np.array(self._data_set['dmb'][this_survey]))


            n_SN_survey.append(sum(this_survey))


            

        self._n_SN_survey = np.array(n_SN_survey)
            

        self._zcmb_survey=np.array(zcmb_survey)

        self._color_survey=np.array(color_survey)

        self._dcolor_survey=np.array(dcolor_survey)
            
        self._x1_survey=np.array(x1_survey)
            
        self._dx1_survey=np.array(dx1_survey)

        self._mbObs_survey=np.array(mbObs_survey)

        self._dmbObs_survey=np.array(dmbObs_survey)
            


        


        


    def compute_advi(self,n_samples=100000, verbose=False):

        
        with self._model:

            self._v_params = pm.variational.advi(n=n_samples, verbose=verbose)

            if verbose:

                fig, ax = plt.subplots()

                ax.plot(-np.log10(-self._v_params.elbo_vals))

        self._advi_complete = True
        
    def sample(self,n_samples=3000,n_jobs=2):


        assert self._advi_complete == True

        with self._model:
    
            step = pm.NUTS(scaling=np.power(self._model.dict_to_array(self._v_params.stds),2), is_cov=True)

            self._trace = pm.sample(draws=n_samples, njobs=n_jobs, step=step, start=self._v_params.means)

    
    @property
    def trace(self):

        if self._trace is not None:

            return self._trace

            


    def plot_data(self):


        fig, ax = plt.subplots()
        for survey in [1,2,3,4]:

            surCondition = self._data_set['set'] == survey

        ax.plot(self._data_set[surCondition]['zcmb'],self._data_set[surCondition]['mb'],'.',label=survey,alpha=.7)
    

        ax.legend(loc=4)

        ax.set_xlabel('redshift (z)')

        ax.set_ylabel(r'magnitude ($\mu$)')



class BaseLineModel(SNBayesModel):

    def _model_setup(self):


        with self._model:    
    
  
    
    
    
    
            # COSMOLOGY
    
    
            OmegaM = pm.Uniform("OmegaM",lower=0,upper=1.)
    
            # Only needed if we want to sample w
            w      = pm.Normal("w",mu=-1,sd=1)
    
    
            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.
    
            dm = distmodW(OmegaM,self._h0,w,self._zcmb)
            #dm = distmod(OmegaM,self._h0,JZ)
  
    
    
            # PHILIPS PARAMETERS
    
            # M0 is the location parameter for the distribution 
            # sysScat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0      = pm.Normal("M0",mu=-19.3,sd=2.)
            sysScat = pm.HalfCauchy('sysScat',beta=2.5) # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true',M0,sysScat,shape=self._n_SN)
    
    
    
            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha",lower=-.2,upper=.3)
            taninv_beta = pm.Uniform("taninv_beta",lower=-1.4,upper=1.4)
    
            # Transform variables
            alpha=pm.Deterministic('alpha',T.tan(taninv_alpha))
            beta=pm.Deterministic('beta',T.tan(taninv_beta))
    
  
            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.
            
            xm=pm.Cauchy('xm',alpha=0,beta=1)
            cm=pm.Cauchy('cm',alpha=0,beta=1)
            
            Rx_log=pm.Uniform('Rx_log',lower=-0.5,upper=0.5)
            Rc_log=pm.Uniform('Rc_log',lower=-1.5,upper=1.5)
    
            # Transformed variables
            Rx=pm.Deterministic("Rx",T.pow(10.,Rx_log))
            Rc=pm.Deterministic("Rc",T.pow(10.,Rc_log))


            x_true  = pm.Normal('x_true',mu=xm, sd=Rx,shape=self._n_SN)
            c_true  = pm.Normal('c_true',mu=cm, sd=Rc,shape=self._n_SN)
    
    
   

            # Do the correction 
            mb = pm.Deterministic("mb",M_true + dm - alpha*x_true + beta*c_true)
    
    
            # Likelihood and measurement error
    
            obsc=pm.Normal("obsc",mu=c_true,sd=self._dcolor, observed=self._color)
            obsx=pm.Normal("obsx",mu=x_true,sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm",mu=mb,sd=self._dmbObs,observed=self._mbObs) 


class HostMassCovariateCorrection(SNBayesModel):

    def _model_setup(self):


        with self._model:    
    
  
    
    
    
    
            # COSMOLOGY
    

    
            OmegaM = pm.Uniform("OmegaM",lower=0,upper=1.)
    
            # Only needed if we want to sample w
            w      = pm.Normal("w",mu=-1,sd=1)
    
    
            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.
    
            dm = distmodW(OmegaM,self._h0,w,self._zcmb)
            #dm = distmod(OmegaM,self._h0,JZ)
  
    
    
            # PHILIPS PARAMETERS
    
            # M0 is the location parameter for the distribution 
            # sysScat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0      = pm.Normal("M0",mu=-19.3,sd=2.)
            sysScat = pm.HalfCauchy('sysScat',beta=2.5) # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true',M0,sysScat,shape=self._n_SN)
    

            Mg0 = pm.Normal("Mg_star",mu=10,sd=100)
            Rg_log = pm.Uniform("Rg_log",lower=-5,upper=2)
            
    
            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha",lower=-.2,upper=.3)
            taninv_beta = pm.Uniform("taninv_beta",lower=-1.4,upper=1.4)
            taninv_gamma = pm.Uniform("taninv_gamma",lower=-1.4,upper=1.4)
    
            # Transform variables
            alpha=pm.Deterministic('alpha',T.tan(taninv_alpha))
            beta=pm.Deterministic('beta',T.tan(taninv_beta))
            gamma=pm.Deterministic('gamma',T.tan(taninv_gamma))
    
  
            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.
            
            xm=pm.Cauchy('xm',alpha=0,beta=1)
            cm=pm.Cauchy('cm',alpha=0,beta=1)
            
            Rx_log=pm.Uniform('Rx_log',lower=-0.5,upper=0.5)
            Rc_log=pm.Uniform('Rc_log',lower=-1.5,upper=1.5)
    
            # Transformed variables
            Rx=pm.Deterministic("Rx",T.pow(10.,Rx_log))
            Rc=pm.Deterministic("Rc",T.pow(10.,Rc_log))
            Rg=pm.Deterministic("Rg",T.pow(10.,Rg_log))


            x_true  = pm.Normal('x_true',mu=xm, sd=Rx,shape=self._n_SN)
            c_true  = pm.Normal('c_true',mu=cm, sd=Rc,shape=self._n_SN)

            Mg_true = pm.Normal('Mg_true',mu=Mg0,sd=Rg,shape=self._n_SN) 
    
   

            # Do the correction 
            mb = pm.Deterministic("mb",M_true + dm - alpha*x_true + beta*c_true+gamma*Mg_true)
    
    
            # Likelihood and measurement error
    
            obsc=pm.Normal("obsc",mu=c_true,sd=self._dcolor, observed=self._color)
            obsx=pm.Normal("obsx",mu=x_true,sd=self._dx1, observed=self._x1)
            obsMg=pm.Normal("obsMg",mu=Mg_true,sd=self._dlog_host_mass, observed=self._log_host_mass)
            obsm = pm.Normal("obsm",mu=mb,sd=self._dmbObs,observed=self._mbObs) 



            
class BaseLineModelWithRedshiftCorrection(SNBayesModel):

    def _model_setup(self):


        with self._model:    
    
  
    
    
    
    
            # COSMOLOGY
    

    
            OmegaM = pm.Uniform("OmegaM",lower=0,upper=1.)
    
            # Only needed if we want to sample w
            w      = pm.Normal("w",mu=-1,sd=1)
    
    
            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.
    
            dm = distmodW(OmegaM,self._h0,w,self._zcmb)
            #dm = distmod(OmegaM,self._h0,JZ)
  
    
    
            # PHILIPS PARAMETERS
    
            # M0 is the location parameter for the distribution 
            # sysScat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0      = pm.Uniform("M0",lower=-20.,upper=-18.)
            sysScat = pm.HalfCauchy('sysScat',beta=2.5) # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true',M0,sysScat,shape=self._n_SN)
    
    
    
            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha",lower=-.2,upper=.3)
            taninv_beta = pm.Uniform("taninv_beta",lower=-1.4,upper=1.4)
    
            # Transform variables
            alpha=pm.Deterministic('alpha',T.tan(taninv_alpha))
            beta=pm.Deterministic('beta',T.tan(taninv_beta))

            # Z correction parameters
            deltaBeta = pm.Uniform('deltaBeta',lower=-1.5,upper=1.5)
            zt = pm.Uniform('zt',lower=0.2,upper=1)
            
  
            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.
            
            xm=pm.Cauchy('xm',alpha=0,beta=1)
            cm=pm.Cauchy('cm',alpha=0,beta=1)

            
            Rx_log=pm.Uniform('Rx_log',lower=-0.5,upper=0.5)
            Rc_log=pm.Uniform('Rc_log',lower=-1.5,upper=1.5)
    
            # Transformed variables
            Rx=pm.Deterministic("Rx",T.pow(10.,Rx_log))
            Rc=pm.Deterministic("Rc",T.pow(10.,Rc_log))


            x_true  = pm.Normal('x_true',mu=xm, sd=Rx,shape=self._n_SN)
            c_true  = pm.Normal('c_true',mu=c_shift, sd=Rc,shape=self._n_SN)
    
    
   

            # Do the correction 
            mb = pm.Deterministic("mb",M_true + dm - alpha*x_true + beta*c_true +deltaBeta*(0.5 + 1./np.pi*T.arctan((self._zcmb - zt)/0.01) )*c_true )
    
    
            # Likelihood and measurement error
    
            obsc=pm.Normal("obsc",mu=c_true,sd=self._dcolor, observed=self._color)
            obsx=pm.Normal("obsx",mu=x_true,sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm",mu=mb,sd=self._dmbObs,observed=self._mbObs) 





class PopulationColorCorrection(SNBayesModel):





    def _model_setup(self):


        with self._model:    
    
  
    
    
    
    
            # COSMOLOGY
    
    
            OmegaM = pm.Uniform("OmegaM",lower=0,upper=1.)
    
            # Only needed if we want to sample w
            w      = pm.Normal("w",mu=-1,sd=1)
    
    
            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.
    


            #  We are going to have to break this into
            #  four likelihoods

            # dm_0 = distmod(OmegaM,self._h0,self._zcmb_survey[0])
            # dm_1 = distmod(OmegaM,self._h0,self._zcmb_survey[1])
            # dm_2 = distmod(OmegaM,self._h0,self._zcmb_survey[2])
            # dm_3 = distmod(OmegaM,self._h0,self._zcmb_survey[3])


            dm_0 = distmodW(OmegaM,self._h0,w,self._zcmb_survey[0])
            dm_1 = distmodW(OmegaM,self._h0,w,self._zcmb_survey[1])
            dm_2 = distmodW(OmegaM,self._h0,w,self._zcmb_survey[2])
            dm_3 = distmodW(OmegaM,self._h0,w,self._zcmb_survey[3])
  
    
    
            # PHILIPS PARAMETERS
    
            # M0 is the location parameter for the distribution 
            # sysScat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0      = pm.Uniform("M0",lower=-20.,upper=-18.)
            sysScat = pm.HalfCauchy('sysScat',beta=2.5) # Gelman recommendation for variance parameter

            M_true_0 = pm.Normal('M_true_0',M0,sysScat,shape=self._n_SN_survey[0])
            M_true_1 = pm.Normal('M_true_1',M0,sysScat,shape=self._n_SN_survey[1])
            M_true_2 = pm.Normal('M_true_2',M0,sysScat,shape=self._n_SN_survey[2])
            M_true_3 = pm.Normal('M_true_3',M0,sysScat,shape=self._n_SN_survey[3])
    
    
    
            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha",lower=-.2,upper=.3)
            taninv_beta = pm.Uniform("taninv_beta",lower=-1.4,upper=1.4)
    
            # Transform variables
            alpha=pm.Deterministic('alpha',T.tan(taninv_alpha))
            beta=pm.Deterministic('beta',T.tan(taninv_beta))
    
  
            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.
            
            xm=pm.Cauchy('xm',alpha=0,beta=1)

            cm=pm.Cauchy('cm',alpha=0,beta=1,shape=4)

            s=pm.Uniform('s',lower=-2,upper=2, shape=4)

            
            c_shift_0 = cm[0] + s[0]*self._zcmb_survey[0]
            c_shift_1 = cm[1] + s[1]*self._zcmb_survey[1]
            c_shift_2 = cm[2] + s[2]*self._zcmb_survey[2]
            c_shift_3 = cm[3] + s[3]*self._zcmb_survey[3]

            


            
            Rx_log=pm.Uniform('Rx_log',lower=-0.5,upper=0.5)
            Rc_log=pm.Uniform('Rc_log',lower=-1.5,upper=1.5,shape=4)
    
            # Transformed variables
            Rx=pm.Deterministic("Rx",T.pow(10.,Rx_log))

            Rc=pm.Deterministic("Rc",T.pow(10.,Rc_log))


            x_true_0  = pm.Normal('x_true_0',mu=xm, sd=Rx,shape=self._n_SN_survey[0])
            c_true_0  = pm.Normal('c_true_0',mu=c_shift_0, sd=Rc[0],shape=self._n_SN_survey[0])
            x_true_1  = pm.Normal('x_true_1',mu=xm, sd=Rx,shape=self._n_SN_survey[1])
            c_true_1  = pm.Normal('c_true_1',mu=c_shift_1, sd=Rc[1],shape=self._n_SN_survey[1])
            x_true_2  = pm.Normal('x_true_2',mu=xm, sd=Rx,shape=self._n_SN_survey[2])
            c_true_2  = pm.Normal('c_true_2',mu=c_shift_2, sd=Rc[2],shape=self._n_SN_survey[2])
            x_true_3  = pm.Normal('x_true_3',mu=xm, sd=Rx,shape=self._n_SN_survey[3])
            c_true_3  = pm.Normal('c_true_3',mu=c_shift_3, sd=Rc[3],shape=self._n_SN_survey[3])
    


            
    
   

            # Do the correction 
            mb_0 = pm.Deterministic("mb_0",M_true_0 + dm_0 - alpha*x_true_0 + beta*c_true_0)
            mb_1 = pm.Deterministic("mb_1",M_true_1 + dm_1 - alpha*x_true_1 + beta*c_true_1)
            mb_2 = pm.Deterministic("mb_2",M_true_2 + dm_2 - alpha*x_true_2 + beta*c_true_2)
            mb_3 = pm.Deterministic("mb_3",M_true_3 + dm_3 - alpha*x_true_3 + beta*c_true_3)
    
    
            # Likelihood and measurement error
    
            obsc_0 = pm.Normal("obsc_0", mu=c_true_0, sd=self._dcolor_survey[0], observed=self._color_survey[0])
            obsx_0 = pm.Normal("obsx_0", mu=x_true_0, sd=self._dx1_survey[0],    observed=self._x1_survey[0])
            obsm_0 = pm.Normal("obsm_0", mu=mb_0,     sd=self._dmbObs_survey[0], observed=self._mbObs_survey[0]) 


            obsc_1 = pm.Normal("obsc_1", mu=c_true_1, sd=self._dcolor_survey[1], observed=self._color_survey[1])
            obsx_1 = pm.Normal("obsx_1", mu=x_true_1, sd=self._dx1_survey[1],    observed=self._x1_survey[1])
            obsm_1 = pm.Normal("obsm_1", mu=mb_1,     sd=self._dmbObs_survey[1], observed=self._mbObs_survey[1]) 


            obsc_2 = pm.Normal("obsc_2", mu=c_true_2, sd=self._dcolor_survey[2], observed=self._color_survey[2])
            obsx_2 = pm.Normal("obsx_2", mu=x_true_2, sd=self._dx1_survey[2],    observed=self._x1_survey[2])
            obsm_2 = pm.Normal("obsm_2", mu=mb_2,     sd=self._dmbObs_survey[2], observed=self._mbObs_survey[2]) 


            obsc_3 = pm.Normal("obsc_3", mu=c_true_3, sd=self._dcolor_survey[3], observed=self._color_survey[3])
            obsx_3 = pm.Normal("obsx_3", mu=x_true_3, sd=self._dx1_survey[3],    observed=self._x1_survey[3])
            obsm_3 = pm.Normal("obsm_3", mu=mb_3,     sd=self._dmbObs_survey[3], observed=self._mbObs_survey[3]) 





        
