#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import array, linspace, meshgrid, pi, zeros, sin, sqrt, arctan2
from numpy import save as npsave
from numpy import load as npload
from numpy.fft import fft, ifft
from scipy.ndimage import geometric_transform as transform
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(
    description="Applies a filter to an image reconstructed from its Polar Broken Ray transform.")
parser.add_argument("--in", "-i", dest="input", action="store",
                    help="Input file. Default: <stdin>.", default="-")
parser.add_argument("--out", "-o", dest="output", action="store",
                    help="Output file. Default: <stdout>.", default="-")
parser.add_argument("--roi", "-R", dest="roi",
                    action="store", help="Specify region of interest. Format: xmin:xmax:ymin:ymax")
parser.add_argument("--size", "-S", dest="size",
                    action="store", help="Specify desired size. Format width:height.")
parser.add_argument("--beta", "-b", dest="beta",
                    action="store", help="Filter parameter beta.", default=None)
parser.add_argument("--overwrite", "-y", dest="overwrite",
                    action='store_const', help="Overwrite output file, if it exists.", const=True)
parser.add_argument("--no-overwrite", "-n", dest="nooverwrite", action='store_const',
                    help="Do not overwrite output file, if it exists.", const=True)
args = parser.parse_args()

msgout = sys.stdout if sys.stdout.isatty() else sys.stderr

if args.overwrite and args.nooverwrite:
    print >>sys.stderr, "Cannot specify both '-y' and '-n'!"
    sys.exit()

if args.input == "-":
    if sys.stdin.isatty():
        print >>sys.stderr, "Surely you are not typing the raw data into the terminal. Please specify input file, or pipe input from another program.\n"
        parser.print_help(sys.stderr)
        sys.exit()
    infile = sys.stdin
else:
    infile = open(args.input, "rb")

if args.output == "-":
    if sys.stdout.isatty():
        print >>sys.stderr, "Cowardly refusing to write binary data to terminal. Please specify output file, or redirect output to a pipe.\n"
        parser.print_help(sys.stderr)
        sys.exit()
    outfile = sys.stdout
    print >>sys.stderr, "Writing data to <stdout>."
else:
    if os.path.exists(args.output):
        if args.nooverwrite:
            print >>sys.stderr, "Error: Output file '%s' exists. Terminating because '-n' was specified." % args.output
            sys.exit()
        elif args.overwrite:
            print >>msgout, "Warning: Output file '%s' exists. Overwriting because '-y' was specified." % args.output
        elif sys.stdin.isatty() and sys.stdout.isatty():
            overwrite = raw_input(
                "Warning: Output file '%s' exists. Do you wish to overwrite file? (Y/N) " % args.output)
            while overwrite.upper() not in ("Y", "N", "YES", "NO"):
                print >>msgout, "Invalid answer: '%s'" % overwrite
                overwrite = raw_input(
                    "Warning: Output file '%s' exists. Do you wish to overwrite file? (Y/N) " % args.output)

            if overwrite.upper() in ("Y", "YES"):
                print >>msgout, "Overwriting '%s'." % args.output
                print >>msgout, ""
            elif overwrite.upper() in ("N", "NO"):
                print >>msgout, "Operation aborted."
                sys.exit()
        else:
            print >>sys.stderr, "Operation aborted. Cowardly refusing to overwrite '%s'." % args.output
            sys.exit()

    outfile = args.output

tag = npload(infile)
metadata = npload(infile)
data = npload(infile)
if infile is not sys.stdin:
    infile.close()

fdata = fft(data, axis=0)

if len(data.shape) != 2:
    print >>sys.stderr, "Expected a two-dimensional array. Got an array with shape %s instead." % (
        data.shape,)
    sys.exit()

kind_str = {
    "f": "floating point",
    "i": "integer",
    "u": "unsigned integer",
    "b": "boolean",
    "c": "complex"
}

if data.dtype.kind in "iub":
    print >>sys.stderr, "Error: Data type '%s' not supported." % kind_str[
        data.dtype.kind]
    sys.exit()

(rmin, rmax, thetamin, thetamax) = metadata
H, W = data.shape

print >>msgout, u"Array information:"
print >>msgout, u"   %.5f ≤ r ≤ %.5f, W = %d, ∆r = %.5f" % \
    (rmin, rmax, W, (rmax - rmin) / (W - 1))
print >>msgout, u"   %.5f ≤ θ ≤ %.5f, H = %d, ∆θ = %.5f" % \
    (thetamin, thetamax, H, (thetamax - thetamin) / (H - 1))
print >>msgout, "Data type: %d-bit %s" % (
    8 * data.dtype.alignment, kind_str[data.dtype.kind])

if args.roi is None:
    xmin = ymin = 0
    xmax = ymax = rmax
else:
    try:
        xmin, xmax, ymin, ymax = args.roi.split(":")
    except:
        print >>sys.stderr, "Bad parameter for roi. Expected colon-delimited list of four floating points or integers, got '%s' instead.\n" % args.roi
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        xmin = float(xmin)
    except:
        print >>sys.stderr, "Bad parameter for xmin. Expected floating point or integer, got '%s' instead.\n" % xmin
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        ymin = float(ymin)
    except:
        print >>sys.stderr, "Bad parameter for ymin. Expected floating point or integer, got '%s' instead.\n" % ymin
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        xmax = float(xmax)
    except:
        print >>sys.stderr, "Bad parameter for xmax. Expected floating point or integer, got '%s' instead.\n" % xmax
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        ymax = float(ymax)
    except:
        print >>sys.stderr, "Bad parameter for ymax. Expected floating point or integer, got '%s' instead.\n" % ymax
        parser.print_help(sys.stderr)
        sys.exit()

if args.size is None:
    W_new = int(640 * sqrt((xmax - xmin) / (ymax - ymin)))
    H_new = int(640 * sqrt((ymax - ymin) / (xmax - xmin)))
else:
    try:
        W_new, H_new = args.size.split(":")
    except:
        print >>sys.stderr, "Bad parameter for size. Expected colon-delimited pair of integers, got '%s' instead.\n" % args.size
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        W_new = int(W_new)
    except:
        print >>sys.stderr, "Bad parameter for width. Expected integer, got '%s' instead.\n" % W_new
        parser.print_help(sys.stderr)
        sys.exit()

    try:
        H_new = int(H_new)
    except:
        print >>sys.stderr, "Bad parameter for width. Expected integer, got '%s' instead.\n" % H_new
        parser.print_help(sys.stderr)
        sys.exit()

dx = (xmax - xmin) / (W_new - 1)
dy = (ymax - ymin) / (H_new - 1)

if args.beta is not None:
    try:
        args.beta = float(args.beta)
    except ValueError:
        print >>sys.stderr, "Bad parameter for beta. Expected floating point or integer, got '%s' instead." % args.beta
        sys.exit()
else:
    args.beta = 2 * sqrt(dx * dy)


print >>msgout, u"Output array information:"
print >>msgout, u"   %.5f ≤ x ≤ %.5f, W = %d, ∆x = %.5f" % \
    (xmin, xmax, W_new, dx)
print >>msgout, u"   %.5f ≤ y ≤ %.5f, H = %d, ∆y = %.5f" % \
    (ymin, ymax, H_new, dy)
print >>msgout, u"Filter parameter: β = %.5f." % args.beta


N = linspace(0, H - 1, H)
N[H / 2 + 1:] -= H
R = linspace(rmin, rmax, W)
R, N = meshgrid(R, N)

mask = (abs(N) <= 2 * pi * R / args.beta) & (N != 0)

mult = zeros(R.shape)
mult[mask] = 2 * pi * R[mask] * \
    sin(args.beta * N[mask] / (2 * pi * R[mask])) / (args.beta * N[mask])
mult[N == 0] = 1

filtered = ifft(fdata * mult, axis=0)

new_metadata = array((xmin, xmax, ymin, ymax))


class ProgressIndicator(object):

    def __init__(self, done, progress=0, msgout=sys.stdout, msg="Progress"):
        self.done = done
        self.progress = 0
        self.msgout = msgout
        self.msg = msg

    def __call__(self, increment=1):
        self.progress += increment
        print >>self.msgout, "%s: %.2f%% done.\r" % (
            self.msg, self.progress * 100.0 / self.done),
        self.msgout.flush()


def polar2cartesian(outcoords, rmin, rmax, thetamin, thetamax, inputshape, xmin, xmax, ymin, ymax, outputshape, progress=None):
    # Coordinate transform for converting a polar array
    # to Cartesian coordinates.
    # 'outputshape' is a tuple containing the shape of the
    # Cartesian array.

    Hout, Wout = outputshape
    Hin, Win = inputshape

    dx = (xmax - xmin) / (Wout - 1)
    dy = (ymax - ymin) / (Hout - 1)

    dr = (rmax - rmin) / (Win - 1)
    dtheta = (thetamax - thetamin) / (Hin - 1)

    yk, xk = outcoords

    x = xmin + xk * dx
    y = ymin + yk * dy

    r = sqrt(x ** 2 + y ** 2)
    if x == 0 and y == 0:
        theta = 0
    else:
        theta = arctan2(y, x)

    thetak = (theta - thetamin) / dtheta
    rk = (r - rmin) / dr

    if progress and callable(progress):
        progress()
    return (thetak, rk)

progress = ProgressIndicator(
    H_new * W_new, msgout=msgout, msg="Reconstructing to Cartesian grid")
kwargs = dict(outputshape=(H_new, W_new), inputshape=filtered.shape, xmin=xmin, xmax=xmax,
              ymin=ymin, ymax=ymax, rmin=rmin, rmax=rmax, thetamin=thetamin, thetamax=thetamax, progress=progress)

t0 = time.time()
new_data = transform(filtered.real, polar2cartesian, order=3, output_shape=(
    H_new, W_new), extra_keywords=kwargs)
print >>msgout, ""
print >>msgout, "Reconstruction done: %.2f seconds" % (time.time() - t0)

t0 = time.time()
print >>msgout, "Writing data to '%s'..." % args.output,
msgout.flush()
if args.output != "-":
    outfile = open(args.output, "wb")
npsave(outfile, "data")
npsave(outfile, new_metadata)
npsave(outfile, new_data)
if outfile is not sys.stdout:
    outfile.close()
print >>msgout, "%.2f seconds" % (time.time() - t0)
