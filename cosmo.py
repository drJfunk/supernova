__author__ = 'drjfunk'

import theano.tensor as T
from theano.ifelse import ifelse

from astropy.constants import c as sol
from astropy import cosmology

from integration_routines import gauss_kronrod


# this sets up the radiation content
cosmo = cosmology.FlatLambdaCDM(H0=67.3, Om0=.3)
Or = cosmo.Onu0 + cosmo.Ogamma0
sol = sol.value


def H_flat(z, Om, h0):
    zp = (1 + z)
    Ode = 1 - Om - Or
    return h0 * T.sqrt(T.pow(zp, 3) * Om + Ode)


def I_int(z, Om):
    zp = (1 + z)
    Ode = 1 - Om - Or
    return  T.power(T.pow(zp, 3) * Om + Ode, -0.5)



def distmod_flat(Om, h0, z):


    dh = sol * 1.e-3 / h0

    dc = dh * gauss_kronrod(I_int, z, parameters=[Om])

    dl = (1 + z) * dc
    return 5. * T.log10(dl) + 25.


# FLAT W


def Hw_flat(z, Om, h0, w):
    zp = (1 + z)
    Ode = 1 - Om - Or
    return h0 * T.sqrt(T.pow(zp, 3) * (Or * zp + Om)
                       + Ode * T.pow(zp, 3. * (1 + w)))


def I_intw(z, Om, w):
    zp = (1 + z)
    Ode = 1 - Om - Or
    return T.power((T.pow(zp, 3) * (Or * zp + Om)
                       + Ode * T.pow(zp, 3. * (1 + w))),-0.5)



def distmod_flat_W(Om, h0, w, z):


    dh = sol * 1.e-3 / h0

    dc = dh * gauss_kronrod(I_intw,z, parameters=[Om, w])
    dl = (1+z) * dc


    return 5. * T.log10(dl) + 25.


# Curvature


def H_curve(z, Om, Ok, h0):
    zp = (1 + z)
    Ode = 1 - Om - Or - Ok
    return T.sqrt(zp * zp * ((Or * zp + Om) * zp + Ok) + Ode)




def I_int_curve(z, Om, Ok):
    zp = (1 + z)
    Ode = 1 - Om - Or - Ok
    return T.power(zp * zp * ((Or * zp + Om) * zp + Ok) + Ode,-0.5)



def distmod_curve(Om, Ok, h0, z):


    dh = sol * 1.e-3 / h0

    dc = dh * gauss_kronrod(I_int_curve, z, parameters=[Om, Ok])



    sqrtOk = T.sqrt(T.abs_(Ok))



    dl =  ifelse(T.eq(Ok,0.),
                 (1+z) * dc,
                 0. * (1+z) * dc)


    dl +=  ifelse(T.gt(Ok,0),
                 (1+z) * dh / sqrtOk * T.sinh(sqrtOk * dc / dh),
                 0. * (1+z) * dc)

    dl += ifelse(T.lt(Ok,0),
                (1+z) * dh / sqrtOk * T.sin(sqrtOk * dc / dh),
                0. * (1+z) * dc)




    return 5. * T.log10(dl) + 25.
