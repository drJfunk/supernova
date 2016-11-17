__author__ = 'drjfunk'

import theano.tensor as T
from theano.ifelse import ifelse
from astropy.constants import c as sol
from astropy import cosmology
from integration_routines import gauss_kronrod


# this sets up the radiation content of the universe.
# It is probably a valid assumption that Planck gets this right
cosmo = cosmology.FlatLambdaCDM(H0=67.3, Om0=.3)
Or = cosmo.Onu0 + cosmo.Ogamma0
sol = sol.value



# Flat universe with cosmological constant


def integrand_constant_flat(z, Om):
    """

    :param z: redshift
    :param Om: matter content
    :return: theano array of 1/H(z)
    """
    zp = (1 + z)
    Ode = 1 - Om - Or # Adjust cosmological constant

    return  T.power(T.pow(zp, 3) * Om + Ode, -0.5)



def distmod_constant_flat(Om, h0, z):
    """
    Distance modulus for a flat universe with a
    cosmological constant

    :param Om: matter content
    :param h0: hubble constant
    :param z: redshift
    :return: theano array of dist. mods.
    """

    # Hubble distance
    dh = sol * 1.e-3 / h0

    # comoving distance
    dc = dh * gauss_kronrod(integrand_constant_flat, z, parameters=[Om])

    # luminosity distance
    dl = (1 + z) * dc

    return 5. * T.log10(dl) + 25. # dist mod.


# Flat universe with dark energy equation of state: w != -1


def integrand_w_flat(z, Om, w):
    """

    :param z: redshift
    :param Om: matter content
    :param w: DE EOS
    :return: theano array of 1/H(z)
    """
    zp = (1 + z)
    Ode = 1 - Om - Or # Adjust cosmological constant
    return T.power((T.pow(zp, 3) * (Or * zp + Om)
                       + Ode * T.pow(zp, 3. * (1 + w))),-0.5)



def distmod_w_flat(Om, h0, w, z):
    """
    Distance modulus for a flat universe with a
    dark energy EOS

    :param Om: matter content
    :param h0: hubble constant
    :param w: DE EOS
    :param z: redshift
    :return: theano array of dist. mods.
    """

    # Hubble distance
    dh = sol * 1.e-3 / h0

    # Comoving distance
    dc = dh * gauss_kronrod(integrand_w_flat, z, parameters=[Om, w])

    # luminosity distance
    dl = (1+z) * dc

    return 5. * T.log10(dl) + 25. #dist mod


# Curved universe with cosmolgical constant


def integrand_constant_curve(z, Om, Ok):
    """

    :param z: redshift
    :param Om: matter content
    :param Ok: curvature
    :return: theano array of 1/H(z)
    """
    zp = (1 + z)
    Ode = 1 - Om - Or - Ok

    return T.power(zp * zp * ((Or * zp + Om) * zp + Ok) + Ode,-0.5)



def distmod_constant_curve(Om, Ok, h0, z):
    """

    Distance modulus for a curved universe with a
    cosmological constant

    :param Om: matter content
    :param Ok: curvature
    :param h0: hubble constant
    :param z: redshift
    :return:  theano array of dist. mods.
    """

    # Hubble distance
    dh = sol * 1.e-3 / h0

    # Comoving distance
    dc = dh * gauss_kronrod(integrand_constant_curve, z, parameters=[Om, Ok])

    # Pre-compute the sqrt
    sqrtOk = T.sqrt(T.abs_(Ok))


    # Theno does not have exhaustive
    # control flow, so we have to compute them all

    # Start here
    dl =  ifelse(T.eq(Ok,0.),
                 (1+z) * dc,
                 0. * (1+z) * dc)

    # The above statement is zero if the
    # condition fails, so we add on to it

    dl +=  ifelse(T.gt(Ok,0),
                 (1+z) * dh / sqrtOk * T.sinh(sqrtOk * dc / dh),
                 0. * (1+z) * dc)

    # same idea as above
    dl += ifelse(T.lt(Ok,0),
                (1+z) * dh / sqrtOk * T.sin(sqrtOk * dc / dh),
                0. * (1+z) * dc)




    return 5. * T.log10(dl) + 25. # dist mod
