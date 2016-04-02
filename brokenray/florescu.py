#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import array, zeros, linspace, meshgrid, ndarray		
from numpy import float64, float128, complex128, complex256
from numpy import exp, sin, cos, tan, arcsin, arctan
from numpy import floor, ceil
from numpy.fft import fft, ifft
from numpy import pi
from numpy import concatenate as concat
from scipy.signal import fftconvolve as conv

import os
import bz2
import rpncalc


class FMSBrokenRayInversion(object):

    """This class defines a series of methods used in the
    reconstruction of a function $\mu_t$ from its Florescu,
    et. al. Broken Ray transform given as an $H \times W$ array, sampling values on
    $\mathrm{ROI}=\left[x_{\min},x_{\max}\right]\times\left[y_{\min},y_{\max}\right]$.
    It will be assumed that $0<\theta\le\frac{\pi}{2}$.
    Scattering angles of $\theta>\frac{\pi}{2}$ are not
    supported."""

    # Initialize 'kernel_generator_v1'. Currently a placeholder
    # that does not actually hold the RPN code needed.
    # Will load RPN from an external file as it is needed.

    dirname, fname = os.path.split(__file__)

    src = "fmsbrt-inv-kernel-placeholder.rpn.bz2"
    f = bz2.BZ2File(os.path.join(dirname, src), "r")
    rpn = f.read().decode("utf8")
    kernel_generator_v1 = rpncalc.decode(rpn)
    f.close()

    # Load both parts of 'kernel_generator_v2'.

    src = "fmsbrt-inv-kernel-v2-part1.rpn.bz2"
    f = bz2.BZ2File(os.path.join(dirname, src), "r")
    rpn = f.read().decode("utf8")
    kernel_generator_v2_part1 = rpncalc.decode(rpn)
    f.close()

    src = "fmsbrt-inv-kernel-v2-part2.rpn.bz2"
    f = bz2.BZ2File(os.path.join(dirname, src), "r")
    rpn = f.read().decode("utf8")
    kernel_generator_v2_part2 = rpncalc.decode(rpn)
    f.close()

    del dirname, fname, src, f, rpn

    def __init__(self, H, W, xmin, xmax, ymin, ymax, theta):
        """Creates an instance of FMSBrokenRayInversion,
        with the following parameters:

        H, W -- The shape of the data.
        xmin, xmax -- The x bounds on the data.
        ymin, ymax -- The y bounds on the data.
        theta -- The fixed scattering angle.
        """
        self.H = H
        self.W = W
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.theta = theta
        if not (0 < self.theta <= pi / 2):
            raise ValueError, \
                "Values of theta outside of (0, pi/2) not supported."
        self.kernel = None

    @property
    def dx(self):
        return float128(self.xmax - self.xmin) / (self.W - 1)

    @property
    def dy(self):
        return float128(self.ymax - self.ymin) / (self.H - 1)

    def _extend(self, data, alpha):
        """Extend data to a larger array needed to perform
        inversion. With an a priori assumption that a function
        $\mu_t$ to be reconstructed is supported in $\mathrm{ROI}$, then
        knowledge of $\mathcal{B}\mu_t$ on $\mathrm{ROI}$ is sufficient, given that $\mathcal{B}\mu_t$
        is then uniquely determined by its values on $\partial\mathrm{ROI}$."""

        # Since an approximation to $\mu_t$ is obtained via convolution
        # of $\mathcal{B}\mu_t$ with $\mathcal{B}^{-1}\phi_{\alpha}$, it is necessary to extend the data
        # to a rectangle $R$ so that $\mathrm{supp}\left(\mathcal{B}^{-1}\phi_{\alpha}\left(z-\cdot\right)\ensuremath{\mathcal{B}\mu_{t}}\right)\subseteq{R}$
        # for each $z\in\mathrm{ROI}$.

        # To reduce the amount of self.* in the code below...
        dx, dy = self.dx, self.dy
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        theta = self.theta
        H, W = self.H, self.W

        # A very quick validation test, that 'data' is an
        # $H \times W$ array.
        if data.shape != (H, W):
            raise ValueError, \
                "data.shape must be (%d, %d)." % (H, W)

        # At a minimium, we take $R=\left[x_{1},x_{2}\right]\times\left[y_{1},y_{\max}\right]$, where:
        #     -- $x_{1}=x_{\min}-\left(\left(\alpha-h\right)\tan\frac{\theta}{2}-\alpha\right)\cos\theta$
        #     -- $x_{2}=x_{\max}+\left(\alpha+h\right)\tan\frac{\theta}{2}+\alpha$
        #     -- $y_{1}=y_{\min}-\left(\alpha\tan\frac{\theta}{2}-\alpha-w\right)\sin\left(\theta\right)$
        #     -- $w=x_{\max}-x_{\min}$
        #     -- $h=y_{\max}-y_{\min}$
        # It is not necessary to extend above the existing $\mathrm{ROI}$.

        ext_left = ((ymax - ymin - alpha)
                    * tan(theta / 2) - alpha) * cos(theta)
        ext_right = (alpha + ymax - ymin)\
            * tan(theta / 2)
        ext_down = (alpha * tan(theta / 2) + alpha
                    + xmax - xmin) * sin(theta)

        # If this module is running inside an instance of SAGE,
        # coerce these variables into float128, because numpy
        # does not play well with SAGE data types.

        xmin_ext = float128(xmin - ext_left)
        xmax_ext = float128(xmax + ext_right)
        ymin_ext = float128(ymin - ext_down)

        W_ext = W + int(ceil(ext_left / dx)
                        + ceil(ext_right / dx))

        H_ext = H + int(ceil(ext_down / dy))

        # Extending to the right is easy.
        # Repeat the boundary data at the right.
        # But we will do that later.

        # Extending to the left and downard takes more effort.
        # We must take boundary data and translate it as it is
        # extended. We use a Fourier multiplier in both
        # extensions to perform the job.

        # Extract left boundary data, pad it in the $y$ direction
        # by $\mathrm{ext}_{\mathrm{left}}\cdot \tan\theta$, then repeat it in the $x$ direction by
        # $\mathrm{ext}_{\mathrm{left}}$.

        pad_up = int(ceil(
            (-(alpha - ymax + ymin) * tan(theta / 2) - alpha) *
            sin(theta) / dy))
        pad_left = int(ceil(ext_left / dx))

        # Left boundary data, padded upward so that we may later
        # throw away the garbage data that will appear when we
        # perform the translation needed to repeat the data
        # in the direction of $-\vec{\mathbf{v}}_2$...
        left = concat((data[::, 0], zeros(pad_up)))
        # ... then repeated to have width 'pad_left'.
        left = array((left,) * pad_left).transpose()

        # We now prepare the Fourier multiplier.
        # Initialize the $x\eta$-grid.

        X = float128(linspace(xmin_ext, xmin - dx,
                              pad_left))

        H_up = H + pad_up
        eta_max = float128(2 * pi * (1 - 1.0 / H_up) / dy)

        Eta = float128(linspace(0, eta_max, H_up))
        Eta[H_up / 2 + 1:] -= float128(2 * pi / dy)

        X, Eta = meshgrid(X, Eta)

        # 'translation' - array giving amount of translation
        # to be performed in the vertical direction.
        translation = float128(tan(theta)) * (X - xmin)

        # The Fourier multiplier used to perform the translation.
        multiplier = exp(-1j * Eta * translation)

        # Note the 'axis=0' keyword argument.
        left_fft = fft(complex128(left), axis=0)
        left = ifft(left_fft * multiplier, axis=0)

        # Concatenate 'left' to 'data', after discarding
        # garbage data.
        data_ext = concat((left[:-pad_up], data), axis=1)

        # We now do the same in extending the data downward.
        pad_right = int(ceil(ext_down * cos(theta) / sin(theta) / dx))
        pad_down = int(ceil(ext_down / dy))

        # Start with the bottom row of the data...
        bottom = concat((data_ext[0], zeros(pad_right)))
        # ...repeated to have height 'pad_down'.
        bottom = array((bottom,) * pad_down)

        # Initialize the ${\xi}y$-grid to prepare the
        # Fourier multiplier.
        Y = float128(linspace(ymin_ext, ymin - dy,
                              pad_down))

        W += pad_right + pad_left
        xi_max = float128(2 * pi * (1 - 1.0 / W) / dx)

        Xi = float128(linspace(0, xi_max, W))
        Xi[W / 2 + 1:] -= float128(2 * pi / dx)

        Xi, Y = meshgrid(Xi, Y)

        translation = float128(cos(theta) / sin(theta)) * (Y - ymin)
        multiplier = exp(-1j * Xi * translation)

        bottom_fft = fft(complex128(bottom), axis=1)
        bottom = ifft(bottom_fft * multiplier, axis=1)

        data_ext = concat((bottom[:, :-pad_right], data_ext),
                          axis=0)

        # Now we extend to the right and return the result.
        pad_right = int(ceil(ext_right / dx))
        data_ext = concat((data_ext,) +
                          (data_ext[:, -1:],) * pad_right, axis=1)

        return data_ext

    def make_kernel(self, alpha, version=1):
        """Prepares a convolution kernel for use to perform numerical
        inversion of the Florescu, et. al. Broken Ray transform."""

        # We now make a convolution kernel $\mathcal{B}^{-1}\phi_{\alpha}$.
        # We must determine a rectangle $S$ needed so that
        # $\mathrm{supp}\left(\ensuremath{\mathcal{B}\mu_{t}}\left(z-\cdot\right)\mathcal{B}^{-1}\phi_{\alpha}\right)\subseteq{S}$ for each $z\in\mathrm{ROI}$.
        # This time, we take $S=\left[x_{1},x_{2}\right]\times\left[y_{1},y_{2}\right]$, where:
        #     -- $x_{1}=\left(-\left(\alpha+h\right)\tan\frac{\theta}{2}-\alpha+w\right)\cos\theta-w$
        #     -- $x_{2}=\alpha+\left(\alpha+h\right)\tan\frac{\theta}{2}$
        #     -- $y_{1}=-h$
        #     -- $y_{2}=\left(\left(\alpha-h\right)\tan\frac{\theta}{2}+\alpha+w\right)\sin\theta+h$
        #     -- $w=x_{\max}-x_{\min}$
        #     -- $h=y_{\max}-y_{\min}$

        dx, dy = self.dx, self.dy
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        theta = self.theta

        # Determine the bounds on the convolution kernel array.
        xmin_ker = (-(alpha + ymax - ymin)
                    * tan(theta / 2)
                    - alpha + xmax - xmin) \
            * cos(theta) \
            - xmax + xmin
        xmax_ker = alpha + (alpha + ymax - ymin) \
            * tan(theta / 2)
        ymin_ker = ymin - ymax
        ymax_ker = ((alpha - ymax + ymin)
                    * tan(theta / 2)
                    + alpha + xmax - xmin) \
            * sin(theta) \
            + ymax - ymin

        # Adjust these bounds to be integer multiples of $\Delta{x}$
        # and $\Delta{y}$ as needed. Also pad around the edge
        # by a single pixel.

        xmin_ker = floor(xmin_ker / dx) * dx
        xmax_ker = ceil(xmax_ker / dx) * dx
        ymin_ker = floor(ymin_ker / dy) * dy
        ymax_ker = ceil(ymax_ker / dy) * dx

        W = int(ceil(xmax_ker / dx) - floor(xmin_ker / dx)) + 1
        H = int(ceil(ymax_ker / dy) - floor(ymin_ker / dy)) + 1

        # Prepare $xy$ grid, with an extra pixel border
        # around the edge.

        X = float128(linspace(xmin_ker - dx, xmax_ker + dx, W + 2))
        Y = float128(linspace(ymin_ker - dy, ymax_ker + dy, H + 2))
        X, Y = meshgrid(X, Y)

        A = cos(float128(theta)) - 1
        B = sin(float128(theta))

        # Prepare $pq$ grid.

        alpha = float128(alpha)

        if version == 1:
            P = B * X - A * Y
            Q = A * X + B * Y

            case_num = self.kernel_generator_v1.findcase(a=A, b=B)
            case, formula = self.kernel_generator_v1[case_num]

            if len(formula) == 0:
                dirname, fname = os.path.split(__file__)
                src = "fmsbrt-inv-kernel-%d.rpn.bz2" % (case_num + 1)

                f = bz2.BZ2File(os.path.join(dirname, src), "r")
                rpn = f.read().decode("utf8")
                f.close()

                self.kernel_generator_v1[case_num] = (case,
                                               rpncalc.decode(rpn))

            Phi = alpha * self.kernel_generator_v1(a=A, b=B,
                                            x=X / alpha, y=Y / alpha,
                                            p=P / alpha, q=Q / alpha)
        elif version == 2:
            W1 = cos(theta/2)
            W2 = sin(theta/2)
            P = -W2*X+W1*Y
            Q = W1*X+W2*Y

            F = self.kernel_generator_v2_part1
            G = self.kernel_generator_v2_part2

            Phi = W1**2/W2*F(x=X/alpha, y=Y/alpha, p=P/alpha,
                            q=Q/alpha, w1=W1, w2=W2)*alpha/4 - \
                          G(x=X/alpha, y=Y/alpha, w1=W1, w2=W2)*alpha*W2/2

        Ker = (Phi[2:] + Phi[:-2] - 2 * Phi[1:-1]) / dx
        Ker = (Ker[:, 2:] + Ker[:, :-2] - 2 * Ker[:, 1:-1]) / dy

        return Ker

    def reconstruct(self, data, kernel, alpha):
        # This is where the reconstruction finally happens.
        data_ext = self._extend(data, alpha)
        result = conv(kernel, data_ext)

        # Need to determine how 'data_ext' is to be cropped.

        dx, dy = self.dx, self.dy
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        theta = self.theta

        # Determine the bounds on the convolution kernel array.
        xmin_ker = (-(alpha + ymax - ymin)
                    * tan(theta / 2)
                    - alpha + xmax - xmin) \
            * cos(theta) \
            - xmax + xmin
        xmax_ker = alpha + (alpha + ymax - ymin) \
            * tan(theta / 2)
        ymin_ker = ymin - ymax
        ymax_ker = ((alpha - ymax + ymin)
                    * tan(theta / 2)
                    + alpha + xmax - xmin) \
            * sin(theta) \
            + ymax - ymin

        # How much the data was extended
        ext_left = ((ymax - ymin - alpha)
                    * tan(theta / 2) - alpha) * cos(theta)
        ext_right = (alpha + ymax - self.ymin)\
            * tan(theta / 2)
        ext_down = (alpha * tan(theta / 2) + alpha
                    + xmax - xmin) * sin(theta)

        crop_bottom = int(ceil(ext_down / dy) - floor(ymin_ker / dy))
        crop_top = int(ceil(ymax_ker / dy))
        crop_left = int(ceil(ext_left / dx) - floor(xmin_ker / dx))
        crop_right = int(ceil(xmax_ker / dx) + ceil(ext_right / dx))

        return result[crop_bottom:-crop_top, crop_left:-crop_right]
