import theano.tensor as T
import theano
import numpy as np

kronrod = np.array([-0.991455371120812639207, -0.949107912342758524526, -0.86486442335976907279,
           -0.7415311855993944398639, -0.5860872354676911302941, -0.4058451513773971669066,
           -0.2077849550078984676007, 0.0, 0.2077849550078984676007, 0.4058451513773971669066,
           0.5860872354676911302941, 0.7415311855993944398639, 0.86486442335976907279,
           0.949107912342758524526, 0.991455371120812639207])

wkronrod = np.array([0.0229353220105292249637, 0.063092092629978553291, 0.10479001032225018384,
            0.140653259715525918745, 0.1690047266392679028266, 0.1903505780647854099133,
            0.204432940075298892414, 0.209482141084727828013, 0.204432940075298892414,
            0.1903505780647854099133, 0.1690047266392679028266, 0.140653259715525918745,
            0.10479001032225018384, 0.063092092629978553291, 0.0229353220105292249637])

gauss = np.array([-0.949107912342758524526, -0.7415311855993944398639, -0.4058451513773971669066,
         0.0, 0.4058451513773971669066, 0.7415311855993944398639, 0.949107912342758524526])

wgauss = np.array([0.129484966168869693271, 0.279705391489276667901, 0.38183005050511894495,
          0.4179591836734693877551, 0.38183005050511894495, 0.279705391489276667901,
          0.129484966168869693271])



# reshape fro proper linear algebra

kronrod.reshape( (1, len(kronrod)) )

wkronrod.reshape( (1, len(wkronrod)) )

gauss.reshape( (1, len(gauss)) )

wgauss.reshape( (1, len(wgauss)) )


# sharing is caring

kronrod_shared = theano.shared(kronrod)

wkronrod_shared = theano.shared(wkronrod)

gauss_shared = theano.shared(gauss)

wgauss_shared = theano.shared(wgauss)




def gauss_kronrod(function, upper_interval, parameters=[]):


    # for the cosmology, a is always 0

    # this is the difference between the interval b-a which is
    # assumed that a is always zero. The same goes for the addition
    # in this case.
    difference = upper_interval.reshape((upper_interval.shape[0], 1))
    #addition   = b.reshape( (b.shape[0], 1) )



    val = T.mul(function(0.5 * T.mul(difference, kronrod_shared) + 0.5 * difference, *parameters), wkronrod_shared)

    return 0.5 * val.sum(axis=1) * difference.T[0]






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


def simpson(f, a, b, n, args=[]):
    h=(b-a)/n
    k=0.0
    x=a + h
    for i in range(1,n/2 + 1):
        k += 4*f(x,*args)
        x += 2*h

    x = a + 2*h
    for i in range(1,n/2):
        k += 2*f(x, *args)
        x += 2*h
    return (h/3)*(f(a,*args)+f(b,*args)+k)
