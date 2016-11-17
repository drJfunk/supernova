__author__ = "J. Michael Burgess"

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import pymc3 as pm
from cosmo import distmod_constant_flat, distmod_w_flat, distmod_constant_curve
from astropy.table import Table


class SNBayesModel(object):
    def __init__(self, sn_table):
        """
        Super class for all supernova models. Variable storage from JLA data sets is done in the
        super class. An astropy table using the JLA parameters must be provided.

        :param sn_table: An astropy Table containing the proper columns w.r.t the JLA data set
        """

        assert isinstance(sn_table, Table), "sn_table must be an astropy Table instance"

        # First we make all the JLA parameters shared to theano.
        # They can be accessed as properties which return numpy
        # arrays

        self._h0 = 67.3  # Planck

        self._zcmb = theano.shared(sn_table['zcmb'])

        self._color = theano.shared(sn_table['color'])

        self._dcolor = theano.shared(sn_table['dcolor'])

        self._x1 = theano.shared(sn_table['x1'])

        self._dx1 = theano.shared(sn_table['dx1'])

        self._mb_obs = theano.shared(sn_table['mb'])

        self._dmb_obs = theano.shared(sn_table['dmb'])

        self._survey = sn_table['set']

        # Just in case there are not host masses in the data set
        # we make sure there is an exception

        try:

            self._log_host_mass = theano.shared(sn_table['3rdvar'])
            self._dlog_host_mass = theano.shared(sn_table['d3rdvar'])

        except:

            self._log_host_mass = None
            self._dlog_host_mass = None

            raise RuntimeWarning("No host galacy mass in data")

        self._n_SN = len(sn_table)

        # For plotting
        self._survey_map = {1: "SNLS", 2: "SDSS", 3: "low-z", 4: "Riess HST"}

        self._data_set = sn_table

        # Some models will want to fit the different surveys with
        # their own parameters, so we create this data set
        self._create_subsurvey_set()

        # generate a pymc3 model
        self._model = pm.Model()

        # The inheriting model will define the setup
        self._model_setup()

        # Just to make sure things are done properly
        self._advi_complete = False

        self._trace = None

    @property
    def zcmb(self):
        return self._zcmb.get_value()

    @property
    def color(self):
        return self._color.get_value()

    @property
    def dcolor(self):
        return self._dcolor.get_value()

    @property
    def x1(self):
        return self._x1.get_value()

    @property
    def dx1(self):
        return self._dx1.get_value()

    @property
    def mb_obs(self):
        return self._mb_obs.get_value()

    @property
    def dmb_obs(self):
        return self._dmb_obs.get_value()

    @property
    def survey_number(self):
        return self._survey

    @property
    def log_host_mass(self):
        return self._log_host_mass.get_value()

    @property
    def dlog_host_mass(self):
        return self._dlog_host_mass.get_value()

    @property
    def data_set(self):
        return self._data_set  # type: Table

    def _model_setup(self):

        print "Super Class"

        pass

    def _create_subsurvey_set(self):
        """
        Creates the sub-survey data sets

        :return: None
        """

        zcmb_survey = []

        color_survey = []

        dcolor_survey = []

        x1_survey = []

        dx1_survey = []

        mbObs_survey = []

        dmbObs_survey = []

        n_SN_survey = []

        self._category = np.array([0, 1, 2, 3])

        for set_number in [1, 2, 3, 4]:

            this_survey = self._data_set['set'] == set_number

            zcmb_survey.append(np.array(self._data_set['zcmb'][this_survey]))

            color_survey.append(np.array(self._data_set['color'][this_survey]))

            dcolor_survey.append(np.array(self._data_set['dcolor'][this_survey]))

            x1_survey.append(np.array(self._data_set['x1'][this_survey]))

            dx1_survey.append(np.array(self._data_set['dx1'][this_survey]))

            mbObs_survey.append(np.array(self._data_set['mb'][this_survey]))

            dmbObs_survey.append(np.array(self._data_set['dmb'][this_survey]))

            n_SN_survey.append(sum(this_survey))

        # I'm not going to share the sub surveys because theano indexing doesn't work the
        # same as numpy

        self._n_SN_survey = np.array(n_SN_survey)

        self._zcmb_survey = np.array(zcmb_survey)

        self._color_survey = np.array(color_survey)

        self._dcolor_survey = np.array(dcolor_survey)

        self._x1_survey = np.array(x1_survey)

        self._dx1_survey = np.array(dx1_survey)

        self._mbObs_survey = np.array(mbObs_survey)

        self._dmbObs_survey = np.array(dmbObs_survey)

    def compute_advi(self, n_samples=100000, verbose=False, **kwargs):

        """

        Compute ADVI of the model to seed the HMC algorithm with
        and efficient step size

        :param n_samples: the number of ADVI samples to compute
        :param verbose: print ADVI steps and plot the ELBO curve
        :param learning_rate: default is 1E-2 but for complex models it is good to increase this
        """
        with self._model:

            self._v_params = pm.variational.advi(n=n_samples, verbose=verbose, **kwargs)

            if verbose:

                fig, ax = plt.subplots()

                ax.plot(-np.log10(-self._v_params.elbo_vals))

        self._advi_complete = True

    def sample(self, n_samples=3000, n_jobs=2):

        """

        Sample the bayesian model after ADVI has been computed

        :param n_samples: number of iteration samples
        :param n_jobs: number of chains to compute
        """
        assert self._advi_complete == True, "must run compute_advi for this model"

        with self._model:
            step = pm.NUTS(scaling=np.power(self._model.dict_to_array(self._v_params.stds), 2), is_cov=True)

            self._trace = pm.sample(draws=n_samples, njobs=n_jobs, step=step, start=self._v_params.means)

    def sample_metropolis(self, n_samples=3000, n_jobs=2):

        """

        Sample the bayesian model after ADVI has been computed

        :param n_samples: number of iteration samples
        :param n_jobs: number of chains to compute
        """
        assert self._advi_complete == True, "must run compute_advi for this model"

        with self._model:
            step = pm.Metropolis()

            self._trace = pm.sample(draws=n_samples, njobs=n_jobs, step=step, start=self._v_params.means)

    @property
    def trace(self):
        """
        Returns the trace after the samples have been computed
        :return: a pymc3 trace
        """

        if self._trace is not None:

            return self._trace

    def plot_data(self, plot_errors=True):
        """
        Plots the distance moduli

        :return: fig
        """

        fig, ax = plt.subplots()
        for survey in [1, 2, 3, 4]:

            survey_condition = self._data_set['set'] == survey

            if plot_errors:

                ax.errorbar(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['mb'],
                            yerr=self._data_set[survey_condition]['dmb'], fmt='.',
                            label=self._survey_map[survey], alpha=.7)

            else:

                ax.plot(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['mb'], '.',
                        label=self._survey_map[survey], alpha=.7)

        ax.legend(loc=4)

        ax.set_xlabel('redshift (z)')

        ax.set_ylabel(r'magnitude ($\mu$)')

        return fig

    def plot_color(self, plot_errors=True):
        """
        Plots the color shift as a function of redshift

        :return: fig
        """

        fig, ax = plt.subplots()
        for survey in [1, 2, 3, 4]:

            survey_condition = self._data_set['set'] == survey

            if plot_errors:

                ax.errorbar(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['color'],
                            yerr=self._data_set[survey_condition]['dcolor'], fmt='.',
                            label=self._survey_map[survey], alpha=.7)

            else:

                ax.plot(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['color'], '.',
                        label=self._survey_map[survey], alpha=.7)

        ax.legend(loc=4)

        ax.set_xlabel('redshift (z)')

        ax.set_ylabel('color')
        ax.set_xscale('log')

        return fig

    def plot_x1(self, plot_errors=True):
        """
        Plots the lightcurve stretch as a function of redshift

        :return: fig
        """

        fig, ax = plt.subplots()
        for survey in [1, 2, 3, 4]:

            survey_condition = self._data_set['set'] == survey

            if plot_errors:

                ax.errorbar(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['x1'],
                            yerr=self._data_set[survey_condition]['dx1'], fmt='.',
                            label=self._survey_map[survey], alpha=.7)

            else:

                ax.plot(self._data_set[survey_condition]['zcmb'], self._data_set[survey_condition]['x1'], '.',
                        label=self._survey_map[survey], alpha=.7)

        ax.legend(loc=4)

        ax.set_xlabel('redshift (z)')

        ax.set_ylabel('x1')
        ax.set_xscale('log')

        return fig

# Baseline models
class BaseLineModelFlat(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC sampling.

            dm = distmod_constant_flat(omega_m, self._h0, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution 
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Normal("M0", mu=-19.3, sd=2.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction 
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)


class BaseLineModelCurvature(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)
            omega_k = pm.Uniform("OmegaK", lower=-1, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC sampling.

            dm = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Normal("M0", mu=-19.3, sd=2.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)


class BaseLineModelFlatW(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # dark energy EOS
            w = pm.Normal("w", mu=-1, sd=1)

            # My custom distance mod. function to enable
            # ADVI and HMC smapling.

            dm = distmod_w_flat(omega_m, self._h0, w, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Normal("M0", mu=-19.3, sd=2.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)


# Host mass models
class HostMassCovariateCorrectionFlat(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY



            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC smapling.

            dm = distmod_constant_flat(omega_m, self._h0, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Normal("M0", mu=-19.3, sd=2.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            Mg0 = pm.Normal("Mg_star", mu=10, sd=100)
            Rg_log = pm.Uniform("Rg_log", lower=-5, upper=2)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)
            taninv_gamma = pm.Uniform("taninv_gamma", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))
            gamma = pm.Deterministic('gamma', T.tan(taninv_gamma))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))
            Rg = pm.Deterministic("Rg", T.pow(10., Rg_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            Mg_true = pm.Normal('Mg_true', mu=Mg0, sd=Rg, shape=self._n_SN)

            # Do the correction
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true + gamma * Mg_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsMg = pm.Normal("obsMg", mu=Mg_true, sd=self._dlog_host_mass, observed=self._log_host_mass)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)


class HostMassCovariateCorrectionFlatW(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY



            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # Dark Energy EOS
            w = pm.Normal("w", mu=-1, sd=1)

            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.

            dm = distmod_w_flat(omega_m, self._h0, w, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution 
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Normal("M0", mu=-19.3, sd=2.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            Mg0 = pm.Normal("Mg_star", mu=10, sd=100)
            Rg_log = pm.Uniform("Rg_log", lower=-5, upper=2)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)
            taninv_gamma = pm.Uniform("taninv_gamma", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))
            gamma = pm.Deterministic('gamma', T.tan(taninv_gamma))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))
            Rg = pm.Deterministic("Rg", T.pow(10., Rg_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            Mg_true = pm.Normal('Mg_true', mu=Mg0, sd=Rg, shape=self._n_SN)

            # Do the correction 
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true + gamma * Mg_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsMg = pm.Normal("obsMg", mu=Mg_true, sd=self._dlog_host_mass, observed=self._log_host_mass)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)

# Redshift Correction models
class BaseLineModelWithRedshiftCorrectionFlat(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY



            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC smapling.


            dm = distmod_constant_flat(omega_m, self._h0, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution 
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Z correction parameters
            delta_beta = pm.Uniform('delta_beta', lower=-1.5, upper=1.5)
            zt = pm.Uniform('zt', lower=0.2, upper=1)

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction 
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true + delta_beta * (
                0.5 + 1. / np.pi * T.arctan((self._zcmb - zt) / 0.01)) * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)

class BaseLineModelWithRedshiftCorrectionCurvature(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY



            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)
            omega_k = pm.Uniform("OmegaK", lower=-1, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC sampling.

            dm = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb)



            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Z correction parameters
            delta_beta = pm.Uniform('delta_beta', lower=-1.5, upper=1.5)
            zt = pm.Uniform('zt', lower=0.2, upper=1)

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true + delta_beta * (
                0.5 + 1. / np.pi * T.arctan((self._zcmb - zt) / 0.01)) * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)

class BaseLineModelWithRedshiftCorrectionFlatW(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY



            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # Dark energy EOS
            w = pm.Normal("w", mu=-1, sd=1)

            # My custom distance mod. function to enable
            # ADVI and HMC smapling.

            dm = distmod_w_flat(omega_m, self._h0, w, self._zcmb)

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter
            M_true = pm.Normal('M_true', M0, sys_scat, shape=self._n_SN)

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Z correction parameters
            delta_beta = pm.Uniform('delta_beta', lower=-1.5, upper=1.5)
            zt = pm.Uniform('zt', lower=0.2, upper=1)

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)
            cm = pm.Cauchy('cm', alpha=0, beta=1)

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))
            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true = pm.Normal('x_true', mu=xm, sd=Rx, shape=self._n_SN)
            c_true = pm.Normal('c_true', mu=cm, sd=Rc, shape=self._n_SN)

            # Do the correction
            mb = pm.Deterministic("mb", M_true + dm - alpha * x_true + beta * c_true + delta_beta * (
                0.5 + 1. / np.pi * T.arctan((self._zcmb - zt) / 0.01)) * c_true)

            # Likelihood and measurement error

            obsc = pm.Normal("obsc", mu=c_true, sd=self._dcolor, observed=self._color)
            obsx = pm.Normal("obsx", mu=x_true, sd=self._dx1, observed=self._x1)
            obsm = pm.Normal("obsm", mu=mb, sd=self._dmb_obs, observed=self._mb_obs)


# Population correction models

class PopulationColorCorrectionFlat(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC sampling.



            #  We are going to have to break this into
            #  four likelihoods

            dm_0 = distmod_constant_flat(omega_m, self._h0, self._zcmb_survey[0])
            dm_1 = distmod_constant_flat(omega_m, self._h0, self._zcmb_survey[1])
            dm_2 = distmod_constant_flat(omega_m, self._h0, self._zcmb_survey[2])
            dm_3 = distmod_constant_flat(omega_m, self._h0, self._zcmb_survey[3])

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter

            M_true_0 = pm.Normal('M_true_0', M0, sys_scat, shape=self._n_SN_survey[0])
            M_true_1 = pm.Normal('M_true_1', M0, sys_scat, shape=self._n_SN_survey[1])
            M_true_2 = pm.Normal('M_true_2', M0, sys_scat, shape=self._n_SN_survey[2])
            M_true_3 = pm.Normal('M_true_3', M0, sys_scat, shape=self._n_SN_survey[3])

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)

            cm = pm.Cauchy('cm', alpha=0, beta=1, shape=4)

            s = pm.Uniform('s', lower=-2, upper=2, shape=4)

            c_shift_0 = cm[0] + s[0] * self._zcmb_survey[0]
            c_shift_1 = cm[1] + s[1] * self._zcmb_survey[1]
            c_shift_2 = cm[2] + s[2] * self._zcmb_survey[2]
            c_shift_3 = cm[3] + s[3] * self._zcmb_survey[3]

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5, shape=4)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))

            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true_0 = pm.Normal('x_true_0', mu=xm, sd=Rx, shape=self._n_SN_survey[0])
            c_true_0 = pm.Normal('c_true_0', mu=c_shift_0, sd=Rc[0], shape=self._n_SN_survey[0])
            x_true_1 = pm.Normal('x_true_1', mu=xm, sd=Rx, shape=self._n_SN_survey[1])
            c_true_1 = pm.Normal('c_true_1', mu=c_shift_1, sd=Rc[1], shape=self._n_SN_survey[1])
            x_true_2 = pm.Normal('x_true_2', mu=xm, sd=Rx, shape=self._n_SN_survey[2])
            c_true_2 = pm.Normal('c_true_2', mu=c_shift_2, sd=Rc[2], shape=self._n_SN_survey[2])
            x_true_3 = pm.Normal('x_true_3', mu=xm, sd=Rx, shape=self._n_SN_survey[3])
            c_true_3 = pm.Normal('c_true_3', mu=c_shift_3, sd=Rc[3], shape=self._n_SN_survey[3])

            # Do the correction
            mb_0 = pm.Deterministic("mb_0", M_true_0 + dm_0 - alpha * x_true_0 + beta * c_true_0)
            mb_1 = pm.Deterministic("mb_1", M_true_1 + dm_1 - alpha * x_true_1 + beta * c_true_1)
            mb_2 = pm.Deterministic("mb_2", M_true_2 + dm_2 - alpha * x_true_2 + beta * c_true_2)
            mb_3 = pm.Deterministic("mb_3", M_true_3 + dm_3 - alpha * x_true_3 + beta * c_true_3)

            # Likelihood and measurement error

            obsc_0 = pm.Normal("obsc_0", mu=c_true_0, sd=self._dcolor_survey[0], observed=self._color_survey[0])
            obsx_0 = pm.Normal("obsx_0", mu=x_true_0, sd=self._dx1_survey[0], observed=self._x1_survey[0])
            obsm_0 = pm.Normal("obsm_0", mu=mb_0, sd=self._dmbObs_survey[0], observed=self._mbObs_survey[0])

            obsc_1 = pm.Normal("obsc_1", mu=c_true_1, sd=self._dcolor_survey[1], observed=self._color_survey[1])
            obsx_1 = pm.Normal("obsx_1", mu=x_true_1, sd=self._dx1_survey[1], observed=self._x1_survey[1])
            obsm_1 = pm.Normal("obsm_1", mu=mb_1, sd=self._dmbObs_survey[1], observed=self._mbObs_survey[1])

            obsc_2 = pm.Normal("obsc_2", mu=c_true_2, sd=self._dcolor_survey[2], observed=self._color_survey[2])
            obsx_2 = pm.Normal("obsx_2", mu=x_true_2, sd=self._dx1_survey[2], observed=self._x1_survey[2])
            obsm_2 = pm.Normal("obsm_2", mu=mb_2, sd=self._dmbObs_survey[2], observed=self._mbObs_survey[2])

            obsc_3 = pm.Normal("obsc_3", mu=c_true_3, sd=self._dcolor_survey[3], observed=self._color_survey[3])
            obsx_3 = pm.Normal("obsx_3", mu=x_true_3, sd=self._dx1_survey[3], observed=self._x1_survey[3])
            obsm_3 = pm.Normal("obsm_3", mu=mb_3, sd=self._dmbObs_survey[3], observed=self._mbObs_survey[3])


class PopulationColorCorrectionCurvature(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0., upper=1.)
            omega_k = pm.Uniform("Omegak", lower=-1., upper=1.)

            # My custom distance mod. function to enable
            # ADVI and HMC sampling.



            #  We are going to have to break this into
            #  four likelihoods

            dm_0 = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb_survey[0])
            dm_1 = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb_survey[1])
            dm_2 = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb_survey[2])
            dm_3 = distmod_constant_curve(omega_m, omega_k, self._h0, self._zcmb_survey[3])

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter

            M_true_0 = pm.Normal('M_true_0', M0, sys_scat, shape=self._n_SN_survey[0])
            M_true_1 = pm.Normal('M_true_1', M0, sys_scat, shape=self._n_SN_survey[1])
            M_true_2 = pm.Normal('M_true_2', M0, sys_scat, shape=self._n_SN_survey[2])
            M_true_3 = pm.Normal('M_true_3', M0, sys_scat, shape=self._n_SN_survey[3])

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead,
            # the Cauchy is less informative around the mean, while
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)

            cm = pm.Cauchy('cm', alpha=0, beta=1, shape=4)

            s = pm.Uniform('s', lower=-2, upper=2, shape=4)

            c_shift_0 = cm[0] + s[0] * self._zcmb_survey[0]
            c_shift_1 = cm[1] + s[1] * self._zcmb_survey[1]
            c_shift_2 = cm[2] + s[2] * self._zcmb_survey[2]
            c_shift_3 = cm[3] + s[3] * self._zcmb_survey[3]

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5, shape=4)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))

            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true_0 = pm.Normal('x_true_0', mu=xm, sd=Rx, shape=self._n_SN_survey[0])
            c_true_0 = pm.Normal('c_true_0', mu=c_shift_0, sd=Rc[0], shape=self._n_SN_survey[0])
            x_true_1 = pm.Normal('x_true_1', mu=xm, sd=Rx, shape=self._n_SN_survey[1])
            c_true_1 = pm.Normal('c_true_1', mu=c_shift_1, sd=Rc[1], shape=self._n_SN_survey[1])
            x_true_2 = pm.Normal('x_true_2', mu=xm, sd=Rx, shape=self._n_SN_survey[2])
            c_true_2 = pm.Normal('c_true_2', mu=c_shift_2, sd=Rc[2], shape=self._n_SN_survey[2])
            x_true_3 = pm.Normal('x_true_3', mu=xm, sd=Rx, shape=self._n_SN_survey[3])
            c_true_3 = pm.Normal('c_true_3', mu=c_shift_3, sd=Rc[3], shape=self._n_SN_survey[3])

            # Do the correction
            mb_0 = pm.Deterministic("mb_0", M_true_0 + dm_0 - alpha * x_true_0 + beta * c_true_0)
            mb_1 = pm.Deterministic("mb_1", M_true_1 + dm_1 - alpha * x_true_1 + beta * c_true_1)
            mb_2 = pm.Deterministic("mb_2", M_true_2 + dm_2 - alpha * x_true_2 + beta * c_true_2)
            mb_3 = pm.Deterministic("mb_3", M_true_3 + dm_3 - alpha * x_true_3 + beta * c_true_3)

            # Likelihood and measurement error

            obsc_0 = pm.Normal("obsc_0", mu=c_true_0, sd=self._dcolor_survey[0], observed=self._color_survey[0])
            obsx_0 = pm.Normal("obsx_0", mu=x_true_0, sd=self._dx1_survey[0], observed=self._x1_survey[0])
            obsm_0 = pm.Normal("obsm_0", mu=mb_0, sd=self._dmbObs_survey[0], observed=self._mbObs_survey[0])

            obsc_1 = pm.Normal("obsc_1", mu=c_true_1, sd=self._dcolor_survey[1], observed=self._color_survey[1])
            obsx_1 = pm.Normal("obsx_1", mu=x_true_1, sd=self._dx1_survey[1], observed=self._x1_survey[1])
            obsm_1 = pm.Normal("obsm_1", mu=mb_1, sd=self._dmbObs_survey[1], observed=self._mbObs_survey[1])

            obsc_2 = pm.Normal("obsc_2", mu=c_true_2, sd=self._dcolor_survey[2], observed=self._color_survey[2])
            obsx_2 = pm.Normal("obsx_2", mu=x_true_2, sd=self._dx1_survey[2], observed=self._x1_survey[2])
            obsm_2 = pm.Normal("obsm_2", mu=mb_2, sd=self._dmbObs_survey[2], observed=self._mbObs_survey[2])

            obsc_3 = pm.Normal("obsc_3", mu=c_true_3, sd=self._dcolor_survey[3], observed=self._color_survey[3])
            obsx_3 = pm.Normal("obsx_3", mu=x_true_3, sd=self._dx1_survey[3], observed=self._x1_survey[3])
            obsm_3 = pm.Normal("obsm_3", mu=mb_3, sd=self._dmbObs_survey[3], observed=self._mbObs_survey[3])


class PopulationColorCorrectionFlatW(SNBayesModel):
    def _model_setup(self):
        with self._model:
            # COSMOLOGY


            omega_m = pm.Uniform("OmegaM", lower=0, upper=1.)

            # Only needed if we want to sample w
            w = pm.Normal("w", mu=-1, sd=1)

            # My custom distance mod. function to enable 
            # ADVI and HMC smapling.



            #  We are going to have to break this into
            #  four likelihoods



            dm_0 = distmod_w_flat(omega_m, self._h0, w, self._zcmb_survey[0])
            dm_1 = distmod_w_flat(omega_m, self._h0, w, self._zcmb_survey[1])
            dm_2 = distmod_w_flat(omega_m, self._h0, w, self._zcmb_survey[2])
            dm_3 = distmod_w_flat(omega_m, self._h0, w, self._zcmb_survey[3])

            # PHILIPS PARAMETERS

            # M0 is the location parameter for the distribution 
            # sys_scat is the scale parameter for the M0 distribution
            # rather than "unexpalined variance"
            M0 = pm.Uniform("M0", lower=-20., upper=-18.)
            sys_scat = pm.HalfCauchy('sys_scat', beta=2.5)  # Gelman recommendation for variance parameter

            M_true_0 = pm.Normal('M_true_0', M0, sys_scat, shape=self._n_SN_survey[0])
            M_true_1 = pm.Normal('M_true_1', M0, sys_scat, shape=self._n_SN_survey[1])
            M_true_2 = pm.Normal('M_true_2', M0, sys_scat, shape=self._n_SN_survey[2])
            M_true_3 = pm.Normal('M_true_3', M0, sys_scat, shape=self._n_SN_survey[3])

            # following Rubin's Unity model... best idea? not sure
            taninv_alpha = pm.Uniform("taninv_alpha", lower=-.2, upper=.3)
            taninv_beta = pm.Uniform("taninv_beta", lower=-1.4, upper=1.4)

            # Transform variables
            alpha = pm.Deterministic('alpha', T.tan(taninv_alpha))
            beta = pm.Deterministic('beta', T.tan(taninv_beta))

            # Again using Rubin's Unity model.
            # After discussion with Rubin, the idea is that
            # these parameters are ideally sampled from a Gaussian,
            # but we know they are not entirely correct. So instead, 
            # the Cauchy is less informative around the mean, while 
            # still having informative tails.

            xm = pm.Cauchy('xm', alpha=0, beta=1)

            cm = pm.Cauchy('cm', alpha=0, beta=1, shape=4)

            s = pm.Uniform('s', lower=-2, upper=2, shape=4)

            c_shift_0 = cm[0] + s[0] * self._zcmb_survey[0]
            c_shift_1 = cm[1] + s[1] * self._zcmb_survey[1]
            c_shift_2 = cm[2] + s[2] * self._zcmb_survey[2]
            c_shift_3 = cm[3] + s[3] * self._zcmb_survey[3]

            Rx_log = pm.Uniform('Rx_log', lower=-0.5, upper=0.5)
            Rc_log = pm.Uniform('Rc_log', lower=-1.5, upper=1.5, shape=4)

            # Transformed variables
            Rx = pm.Deterministic("Rx", T.pow(10., Rx_log))

            Rc = pm.Deterministic("Rc", T.pow(10., Rc_log))

            x_true_0 = pm.Normal('x_true_0', mu=xm, sd=Rx, shape=self._n_SN_survey[0])
            c_true_0 = pm.Normal('c_true_0', mu=c_shift_0, sd=Rc[0], shape=self._n_SN_survey[0])
            x_true_1 = pm.Normal('x_true_1', mu=xm, sd=Rx, shape=self._n_SN_survey[1])
            c_true_1 = pm.Normal('c_true_1', mu=c_shift_1, sd=Rc[1], shape=self._n_SN_survey[1])
            x_true_2 = pm.Normal('x_true_2', mu=xm, sd=Rx, shape=self._n_SN_survey[2])
            c_true_2 = pm.Normal('c_true_2', mu=c_shift_2, sd=Rc[2], shape=self._n_SN_survey[2])
            x_true_3 = pm.Normal('x_true_3', mu=xm, sd=Rx, shape=self._n_SN_survey[3])
            c_true_3 = pm.Normal('c_true_3', mu=c_shift_3, sd=Rc[3], shape=self._n_SN_survey[3])

            # Do the correction 
            mb_0 = pm.Deterministic("mb_0", M_true_0 + dm_0 - alpha * x_true_0 + beta * c_true_0)
            mb_1 = pm.Deterministic("mb_1", M_true_1 + dm_1 - alpha * x_true_1 + beta * c_true_1)
            mb_2 = pm.Deterministic("mb_2", M_true_2 + dm_2 - alpha * x_true_2 + beta * c_true_2)
            mb_3 = pm.Deterministic("mb_3", M_true_3 + dm_3 - alpha * x_true_3 + beta * c_true_3)

            # Likelihood and measurement error

            obsc_0 = pm.Normal("obsc_0", mu=c_true_0, sd=self._dcolor_survey[0], observed=self._color_survey[0])
            obsx_0 = pm.Normal("obsx_0", mu=x_true_0, sd=self._dx1_survey[0], observed=self._x1_survey[0])
            obsm_0 = pm.Normal("obsm_0", mu=mb_0, sd=self._dmbObs_survey[0], observed=self._mbObs_survey[0])

            obsc_1 = pm.Normal("obsc_1", mu=c_true_1, sd=self._dcolor_survey[1], observed=self._color_survey[1])
            obsx_1 = pm.Normal("obsx_1", mu=x_true_1, sd=self._dx1_survey[1], observed=self._x1_survey[1])
            obsm_1 = pm.Normal("obsm_1", mu=mb_1, sd=self._dmbObs_survey[1], observed=self._mbObs_survey[1])

            obsc_2 = pm.Normal("obsc_2", mu=c_true_2, sd=self._dcolor_survey[2], observed=self._color_survey[2])
            obsx_2 = pm.Normal("obsx_2", mu=x_true_2, sd=self._dx1_survey[2], observed=self._x1_survey[2])
            obsm_2 = pm.Normal("obsm_2", mu=mb_2, sd=self._dmbObs_survey[2], observed=self._mbObs_survey[2])

            obsc_3 = pm.Normal("obsc_3", mu=c_true_3, sd=self._dcolor_survey[3], observed=self._color_survey[3])
            obsx_3 = pm.Normal("obsx_3", mu=x_true_3, sd=self._dx1_survey[3], observed=self._x1_survey[3])
            obsm_3 = pm.Normal("obsm_3", mu=mb_3, sd=self._dmbObs_survey[3], observed=self._mbObs_survey[3])
