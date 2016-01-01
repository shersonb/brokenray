# brokenray
Broken Ray transform tools

Provides numerical inversions for the Florescu, Markel, and Schotland Broken Ray transform,
[1], and the Polar Broken Ray transform, [2].

The brokenray module provides two submodules, florescu and polar, and can also be invoked at
the command line.

Also provided are some SAGE worksheets. The worksheet 'fmsbrt-stage1.sws' is used to generate
the piecewise-defined symbolic formulas used in the inversion of the Florescu, et. al. Broken
Ray transform. The resulting *.rpn.bz2 files are provided in this package. The other SAGE
worksheets are data generators, used to simulate data for use as input to the inversions of
both the Florescu, et. al. and Polar BRTs.

The SAGE worksheets require the rpncalc (provided separately) and sageextras modules be placed
in the site-packages directory for the Python installation at

$SAGE_ROOT/local/lib/python/site-packages/,

where $SAGE_ROOT is where your installation of SAGE resides. Additionally, the brokenray
module itself requires numpy, scipy, and rpncalc. The numpy and scipy modules can be obtained
from http://scipy.org/, or through your package manager, whereas the rpncalc module is found
at http://github.com/shersonb/python-rpncalc/. Inversion of the Polar Broken Ray transform also
requires the parallel and iqueue modules, both found at
https://github.com/shersonb/python-parallel/ and https://github.com/shersonb/python-iqueue.

While not a dependency, one should consider using array2im to convert the resulting
reconstructions to image files. The array2im package is found at
https://github.com/shersonb/python-array2im/.

The file 'polar-to-cartesian.py' is also provided as a standalone script used to perform
filtering and resampling the data resulting from the Polar BRT from a polar grid to a
Cartesian grid.

1. Lucia Florescu, Vadim A. Markel, and John C. Schotland, Inversion formulas for the broken-
ray Radon transform, Inverse Problems 27, 025002 (2011) (2010).
2. Brian Sherson, Some Results in Single-Scattering Tomography, Ph.D. thesis, Oregon State
University, December 2015.
