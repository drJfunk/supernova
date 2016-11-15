__author__ ='drjfunk'

import theano.tensor as T
from astropy.constants import c as sol
from astropy import cosmology
import numpy as np






cosmo = cosmology.FlatLambdaCDM(H0=72., Om0=.3)
Oo = cosmo.Onu0 + cosmo.Ogamma0
sol = sol.value
N = 10

loop = range(1, N)


def trapezoidal(f, a, b, n, args=[]):
    h = (b - a) / n

    s = 0.0
    s += f(a, *args) / 2.0
    for i in loop:
        s += f(a + i * h, *args)
    s += f(b, *args) / 2.0
    return s * h


def H(z, Om, h0):
    # h0=72.
    zp = (1 + z)
    # Om=.3
    Ode = 1 - Om - Oo
    return h0 * T.sqrt(T.pow(zp, 3) * Om + Ode)


def I_int(z, Om, h0):
    return sol * 1E-3 / H(z, Om, h0)


def distmod(Om, h0, z):
    # return (1+z) * quad(I_int,0,z)[0]
    dl = (1 + z) * trapezoidal(I_int, 0., z, N, args=[Om, h0])
    return 5. * T.log10(dl) + 25.


def Hw(z, Om, h0, w):
    # h0=72.
    zp = (1 + z)
    # Om=.3
    Ode = 1 - Om - Oo
    return h0 * T.sqrt(T.pow(zp, 3) * (Oo * zp + Om)
                       + Ode * T.pow(zp, 3. * (1 + w)))


def I_intw(z, Om, h0, w):
    return sol * 1E-3 / Hw(z, Om, h0, w)


def distmodW(Om, h0, w, z):
    # return (1+z) * quad(I_int,0,z)[0]
    dl = (1 + z) * trapezoidal(I_intw, 0., z, N, args=[Om, h0, w])
    return 5. * T.log10(dl) + 25.
