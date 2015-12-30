import numpy
from PIL import Image

try:
    import sage
except:
    raise ImportError, "This module is meant to be run from within SAGE."

from sage.rings.integer import Integer
from sage.symbolic.all import assume
from sage.symbolic.constants import *

"""Some extra functions to supplement SAGE."""


def intervals(l, unbounded=False):
    # If $l=\left[x_1,x_2,\dots,x_n\right],
    # returns $\left[\left(x_{1},x_{2}\right),\left(x_{2},x_{3}\right),\dots,\left(x_{n-1},x_{n}\right)\right]$
    # if unbounded is false, otherwise, returns
    # $\left[\left(-\infty,x_{1}\right),\left(x_{1},x_{2}\right),\left(x_{2},x_{3}\right),\dots,\left(x_{n-1},x_{n}\right),\left(x_{n},\infty\right)\right]$
    if unbounded:
        return zip([-infinity] + l, l + [infinity])
    else:
        return zip(l[:-1], l[1:])


def hornerpoly(p, x, *args):
    if type(p) == Integer:
        return p
    terms = p.coefficients(x)

    coeffs, powers = zip(*terms)

    maxpower = max(powers)
    if maxpower == 0:
        if len(args):
            return hornerpoly(p, *args)
        else:
            return p

    q = 0
    for coeff, p1, p2 in zip(coeffs[::-1], powers[::-1], powers[-2::-1] + (0,)):
        if len(args):
            q += hornerpoly(coeff, *args)
        else:
            q += coeff
        q *= x ** (p1 - p2)
    return q


def hornerrational(p, *args):
    num, den = p.numerator_denominator()
    return hornerpoly(num, *args) / hornerpoly(den, *args)


def assume2(*ineqs):
    assumptions = []
    for ineq in ineqs:
        rhs = ineq.rhs()
        lhs = ineq.lhs()
        op = ineq.operator()
        num = (lhs - rhs).numerator()
        den = (lhs - rhs).denominator()
        if bool(den > 0):
            try:
                assume(op(num.expand(), 0))
            except ValueError, msg:
                if msg == "Assumption is redundant":
                    continue
            assumptions.append(op(num.expand(), 0))
        elif bool(den < 0):
            try:
                assume(op(0, num.expand()))
            except ValueError, msg:
                if msg == "Assumption is redundant":
                    continue
            assumptions.append(op(0, num.expand()))
        else:
            raise BaseException, "It would be helpful to know if the denominator '%s' is positive or negative." % den
    return assumptions


def cmp_symbolic(f, g):
    if f == g:
        return int(0)
    elif bool(f < g):
        return int(-1)
    elif bool(f > g):
        return int(1)
    elif bool((f - g).simplify_full() < 0):
        return int(-1)
    elif bool((f - g).simplify_full() > 0):
        return int(1)
    elif bool((f - g).expand() < 0):
        return int(-1)
    elif bool((f - g).expand() > 0):
        return int(1)
    elif bool((f - g).factor() < 0):
        return int(-1)
    elif bool((f - g).factor() > 0):
        return int(1)
    elif bool((f - g).numerator() < 0) and bool((f - g).denominator() > 0):
        return int(-1)
    elif bool((f - g).numerator() > 0) and bool((f - g).denominator() < 0):
        return int(-1)
    elif bool((f - g).numerator() < 0) and bool((f - g).denominator() < 0):
        return int(1)
    elif bool((f - g).numerator() > 0) and bool((f - g).denominator() > 0):
        return int(1)
    else:
        raise BaseException, "Can't compare '%s' and '%s'." % (f, g)


def makegrid2d(H, W, xmin, xmax, ymin, ymax):
    X = numpy.linspace(xmin, xmax, W)
    Y = numpy.linspace(ymin, ymax, H)
    return numpy.meshgrid(X, Y)

_integral_cache = {}


def multivar_integral(integrand, *limits):
    if integrand in _integral_cache.keys():
        cache = _integral_cache[integrand]
    else:
        cache = _integral_cache[integrand] = {}

    if limits not in cache.keys():
        if len(limits) > 1:
            I = multivar_integral(
                integrand, *limits[:-1]).integral(*limits[-1])
        else:
            I = integrand.integral(*limits[0])
        cache[limits] = I

    return cache[limits]
