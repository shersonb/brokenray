#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import array, zeros, linspace, meshgrid, ndarray, diag
from numpy import uint8, float64, int8, int0, float128, complex128
from numpy import exp, sqrt, cos, tan, arctan
from numpy import minimum, maximum
from numpy import ceil, floor
from numpy import matrix as npmatrix
from numpy.fft import fft, ifft
from numpy import pi
from scipy.linalg import solve_triangular as solve
from scipy.signal import fftconvolve as conv
from scipy.ndimage import geometric_transform as transform

# We will make use of *reentrant* locks.
from threading import RLock as Lock
from threading import Condition, Thread

# This module is a modification on python's queue module,
# which allows one to interrupt a queue.
import iqueue

# This is a module written to execute code in parallel.
# While python is limited by the Global Interpreter Lock,
# numerical operations on NumPy arrays are generally not
# limited by the GIL.
import parallel

# This module allows the conversion of SAGE symbolic expressions
# to RPN code through the symbolic_to_rpn. RPNProgram is a subclass
# of list that comes equipped with a __call__ method that implements
# execution of the RPN code.
import rpncalc

def _E(m):
    return int0(npmatrix(diag((1,) * int(m + 1), k=0)[:, :-1]))


def _X(m):
    return int0(npmatrix(diag((1,) * int(m), k=-1)[:, :-1]))


def _Del(m):
    return int0(npmatrix(diag(xrange(1, int(m)), k=1)[:-1]))


class _CD_RPN:

    def __init__(self):
        self.coeffs = [(npmatrix((-1,)), npmatrix((-1,)))]
        self.rpn = [(rpncalc.RPNProgram([-1]), rpncalc.RPNProgram([-1]))]

        # In case this class is utilized by multiple threads.
        self.lock = Lock()

    def getcoeffs(self, m):
        # Returns coefficients for $c_{m}$ and $d_{m}$.
        # If they already exist in cache, just return what is there.
        with self.lock:
            if len(self.coeffs) <= m:
                # Need to generate coefficients for $c_{m}$ and $d_{m}$.
                # Fetch the coefficients for $c_{m-1}$ and $d_{m-1}$.
                C, D = self.getcoeffs(m - 1)

                if m % 2:  # $m$ is odd
                    C_new = _E(m) * D * _X((m + 1) / 2).transpose() \
                        - ((1 + m) * _E(m) + 3 * _X(m)
                           + 2 * (_E(m) + _X(m)) * _X(m - 1) * _Del(m)) * C \
                        * _E((m + 1) / 2).transpose()
                    D_new = _X(m) * C - (m * _E(m) + 2 * _X(m)
                                         + 2 * (_E(m) + _X(m)) * _X(m - 1) * _Del(m)) * D

                else:  # $m$ is even
                    C_new = _E(m) * D * _X(m / 2).transpose() \
                        - ((1 + m) * _E(m) + 3 * _X(m)
                           + 2 * (_E(m) + _X(m)) * _X(m - 1) * _Del(m)) * C

                    D_new = _X(m) * C - (m * _E(m) + 2 * _X(m)
                                         + 2 * (_E(m) + _X(m)) * _X(m - 1) * _Del(m)) * D \
                        * _E(m / 2).transpose()

                self.coeffs.append((C_new, D_new))

            return self.coeffs[m]

    def __getitem__(self, m):
        n2 = rpncalc.wild("n2")
        v2 = rpncalc.wild("v2")
        mul = rpncalc.rpn_funcs[u"⋅"]
        add = rpncalc.rpn_funcs[u"+"]
        # Returns RPN code for $c_j$ and $d_j$. Generate on the fly if needed.
        with self.lock:
            while len(self.rpn) <= m:
                cm_rpn = []
                dm_rpn = []

                C, D = self.getcoeffs(len(self.rpn))

                # Generate RPN code for $c_j$ and $d_j$.
                for row in array(C[::-1]):
                    npoly_rpn = []
                    for coeff in row[::-1]:
                        if coeff:
                            if len(npoly_rpn):
                                npoly_rpn.extend([n2, mul])
                                npoly_rpn.extend([coeff, add])
                            else:
                                npoly_rpn.append(coeff)
                        elif len(npoly_rpn):
                            npoly_rpn.extend([n2, mul])
                    if len(cm_rpn):
                        cm_rpn.extend([v2, mul])
                        cm_rpn.extend(npoly_rpn)
                        cm_rpn.append(add)
                    else:
                        cm_rpn.extend(npoly_rpn)

                for row in array(D[::-1]):
                    npoly_rpn = []
                    for coeff in row[::-1]:
                        if coeff:
                            if len(npoly_rpn):
                                npoly_rpn.extend([n2, mul])
                                npoly_rpn.extend([coeff, add])
                            else:
                                npoly_rpn.append(coeff)
                        elif len(npoly_rpn):
                            npoly_rpn.extend([n2, mul])
                    if len(dm_rpn):
                        dm_rpn.extend([v2, mul])
                        dm_rpn.extend(npoly_rpn)
                        dm_rpn.append(add)
                    else:
                        dm_rpn.extend(npoly_rpn)
                self.rpn.append(
                    (rpncalc.RPNProgram(cm_rpn), rpncalc.RPNProgram(dm_rpn)))
            return self.rpn[m]


class Sderiv:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, A, ds):
        H, W = A.shape
        psi = rpncalc.decode(u"« x 3 ^ 4 / +/- 3 x * 4 / + »")
        N = ceil(self.alpha / ds)
        X = linspace(-N * ds - ds, N * ds + ds, 2 * N + 3)

        Psi = psi(x=X / self.alpha)
        Psi[X > self.alpha] = psi(x=1)
        Psi[X < -self.alpha] = psi(x=-1)

        stencil = (Psi[:-2] + Psi[2:] - 2 * Psi[1:-1]) / ds

        diff = conv([stencil], A)

        return N, N, diff[:, 2 * N:-2 * N]


class PolarBrokenRayInversion(parallel.BaseTaskClass):
    _cd = _CD_RPN()
    _u = rpncalc.decode(u"« q phi sin ⋅ arcsin »")
    _v = rpncalc.decode(u"« q phi sin ⋅ +/- q 2 ^ phi sin 2 ^ ⋅ +/- 1 + √ ÷ »")
    _w = rpncalc.decode(u"« i phi u - ⋅ exp »")
    _tm = rpncalc.decode(u"« i dm ⋅ n ⋅ cm v ⋅ + dlnr m ^ ⋅ m 2 + ! ÷ »")
    _cf = rpncalc.decode(u"« dr r ⋅ v 2 ^ ⋅ phi csc ⋅ s 2 ^ ÷ »")
    _invlock = Lock()

    def __init__(self, Qf, Phi, smin, smax, alpha, nmax=200):
        # Parameters:
        # $\mathbf{Qf}$ -- $\mathcal{Q}f$, sampled on an $r\theta$ grid.
        # $\mathbf{Phi}$ ($\phi$) -- Scattering angle
        # $\mathbf{rmin}$ -- $r_{\min}$, defaults to $1$.
        # $\mathbf{rmax}$ -- $r_{\max}$, defaults to $6$.
        # $\mathbf{D}$ -- Numerical implemenation of $\frac{\partial}{\partial r}$.
        # $\mathbf{nmax}$ -- $n_{\max}$, reconstructs $\tilde{f}\left(r,n\right)$
        # for $\left|n\right| \le n_{\max}$. Defaults to $200$.

        # This reconstruction will assume that $\mathcal{Q}f$ is real and exploit
        # conjugate symmetry in the Fourier series.

        # Initialize variables.
        self.Qf = Qf
        self.Phi = Phi
        self.smin = smin
        self.smax = smax

        H, W = Qf.shape

        self.thetamin = thetamin = -pi
        self.thetamax = thetamax = pi*(1-2.0/H)
        self.nmax = nmax

        self.F = None
        self.F_cartesian = None

        self.lock = Lock()
        self.status = Condition(self.lock)
        self.jobsdone = 0
        self.jobcount = nmax + 1
        self.running = False
        self.projectioncount = 0
        self.projecting = False

        self.dr = dr = ds = (smax - smin) / float(W - 1)
        self.dtheta = dtheta = (thetamax - thetamin) / float(H)

        # Compute $\widetilde{\mathcal{Q}f}$.
        self.FQf = FQf = fft(Qf, axis=0)

        # Perform differentiation of $\widetilde{\mathcal{Q}f}$.
        D = Sderiv(alpha)
        try:
            clip_left, clip_right, self.DFQf = D(FQf, ds)
        except:
            clip_left, clip_right, self.DFQf = D(float64(FQf), ds)

        # Initialize array that will store $\tilde{f}$.
        self.Ff = zeros(self.DFQf.shape, dtype=complex128)

        # Initialize $rs$ grid.
        self.rmin = self.smin + clip_left * ds
        self.rmax = self.smax - clip_right * ds
        R = linspace(self.rmin, self.rmax, W - clip_left - clip_right)
        self.R, self.S = meshgrid(R, R)

        # Compute $q$, $u$, $v$, $w$, and $v^{2}r*\csc(\phi)*{\Delta}r/s^2$.
        self.Q = self.S / self.R

        args = dict(q=self.Q, r=self.R, s=self.S, phi=self.Phi, dr=dr)
        args["u"] = self.U = self._u(**args)
        args["v"] = self.V = self._v(**args)
        self.W = self._w(**args)
        self.Factor = self._cf(**args)

    def A(self, n, eps=0.0000001, p=16):
        # Compute matrix $\mathbf{A}_n$.

        H, W = self.DFQf.shape

        # Initialize the An matrix (as an array for now).
        An = zeros(self.R.shape, dtype=complex128)

        # First compute a partial sum for the upper triangular part.
        # Start with $m=0$

        mask = self.S < self.R

        Sum = zeros(self.R.shape, dtype=complex128)

        for m in xrange(0, p + 1, 2):
            cm_rpn, dm_rpn = self._cd[m]
            Term = self._tm(v=self.V[mask], v2=self.V[mask] ** 2,
                            dlnr=self.dr / self.R[mask],
                            n=n, n2=n ** 2, m=m, cm=cm_rpn, dm=dm_rpn)

            Sum[mask] += Term
            mask[mask] *= abs(Term) >= eps
            if not mask.any():
                break

        mask = self.S < self.R

        An[mask] = 2 * self.W[mask] ** n * self.Factor[mask] * Sum[mask]

        # Now to do the diagonal.
        # Since $r=s$ here, we have $q=1$, $u=\phi$, $v=-\tan\phi$,
        # and $w=1$.

        mask = self.S == self.R

        Sum = zeros(self.R.shape, dtype=complex128)

        for m in xrange(0, p + 1):
            cm_rpn, dm_rpn = self._cd[m]

            Term = self._tm(v=-tan(self.Phi), v2=tan(self.Phi) ** 2,
                            dlnr=self.dr / self.R[mask],
                            n=n, n2=n ** 2, m=m, cm=cm_rpn, dm=dm_rpn)

            Sum[mask] += Term
            mask[mask] *= abs(Term) >= eps
            if not mask.any():
                break

        mask = self.S == self.R
        An[mask] = self.Factor[mask] * Sum[mask] + \
            array([1 - 1 / cos(self.Phi)] * W)

        return npmatrix(An)

    def f(self, n):
        # This is the function that is run in parallel.
        An = self.A(n, eps=10 ** -9, p=24)

        DFQf = self.DFQf[n]

        #AnInv = inv(An).transpose()
        #Ff = array(DFQf*AnInv)[0]
        Ff = solve(An, DFQf)
        return Ff

    def populatequeue(self, queue):
        for n in xrange(self.nmax + 1):
            queue.put(n)

    def postproc(self, (n, Ff)):
        with self.status:
            self.Ff[n] = Ff
            if n > 0:
                self.Ff[-n] = Ff.conjugate()

            self.jobsdone += 1
            self.status.notifyAll()

    def reconstruct(self):
        with self.lock:
            self.F = ifft(self.Ff, axis=0)
            return self.F
