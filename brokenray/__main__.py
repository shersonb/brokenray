#!/usr/bin/env python
import argparse
import os
import sys
import time
from numpy import save as npsave
from numpy import load as npload
from numpy import array, pi, sqrt
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Performs numerical reconstructions of functions from its Broken Ray transform.")
parser.add_argument("--in", "-i", dest="input", action="store",
                    help="Input file. Default: <stdin>", default="-")
parser.add_argument("--out", "-o", dest="output", action="store",
                    help="Output file. Default: <stdout>", default="-")
parser.add_argument("--alpha", "-a", dest="alpha", action="store",
                    help=u"Reconstruction parameter 'alpha'. Higher values of alpha help cancel out error at the cost of blurring the reconstruction. Default: 3⋅√(∆x⋅∆y), or about 6 pixels wide.", default=None)
parser.add_argument("--nmax", "-N", dest="nmax", action="store",
                    help=u"Reconstruction parameter 'nmax' for the Polar Broken Ray transform. Reconstruct Fourier coefficients up to 'nmax'.", default=None)
parser.add_argument("--threads", "-T", dest="threads", action="store",
                    help=u"Number of threads to use (Polar Broken Ray transform only).", default=-1)
parser.add_argument("--overwrite", "-y", dest="overwrite",
                    action='store_const', help="Overwrite output file, if it exists.", const=True)
parser.add_argument("--no-overwrite", "-n", dest="nooverwrite", action='store_const',
                    help="Do not overwrite output file, if it exists.", const=True)
args = parser.parse_args()

msgout = sys.stdout if sys.stdout.isatty() else sys.stderr

if args.overwrite and args.nooverwrite:
    print >>sys.stderr, "Cannot specify both '-y' and '-n'!"
    parser.print_help(sys.stderr)
    sys.exit()

if args.input == "-":
    if sys.stdin.isatty():
        print >>sys.stderr, "Surely you are not typing the raw data into the terminal. Please specify input file, or pipe input from another program."
        print >>sys.stderr, ""
        parser.print_help(sys.stderr)
        sys.exit()
    infile = sys.stdin
else:
    infile = open(args.input, "rb")

if args.output == "-":
    if sys.stdout.isatty():
        print >>sys.stderr, "Cowardly refusing to write binary data to terminal. Please specify output file, or redirect output to a pipe."
        print >>sys.stderr, ""
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
            print >>msgout, ""
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


brt_type = npload(infile)
metadata = npload(infile)

t0 = time.time()
if args.input == "-":
    print >>msgout, "Reading data from <stdin>...",
else:
    print >>msgout, "Reading data from '%s'..." % args.input,

msgout.flush()
data = npload(infile)
if infile is not sys.stdin:
    infile.close()
print >>msgout, "%.2f seconds" % (time.time() - t0)

if len(data.shape) != 2:
    print >>sys.stderr, "Expected a two-dimensional array. Got an array with shape %s instead." % data.shape
    sys.exit()

H, W = data.shape

print >>msgout, ""

kind_str = {"f": "floating point",
            "i": "integer",
            "u": "unsigned integer",
            "b": "boolean",
            "c": "complex"
            }

if data.dtype.kind in "iub":
    print >>sys.stderr, "Error: Data type '%s' not supported." % kind_str[
        data.dtype.kind]
    sys.exit()


if brt_type == "FBRT":
    from florescu import FMSBrokenRayInversion
    try:
        xmin, xmax, ymin, ymax, theta = metadata
    except ValueError:
        print >>sys.stderr, "Unexpected end of data!"
        raise

    print >>msgout, "Inversion of the Florescu, Markel, and Schotland Broken Ray transform."
    print >>msgout, u"Array information:"
    print >>msgout, u"   %.5f ≤ x ≤ %.5f, W = %d, ∆x = %.5f" % \
        (xmin, xmax, W, (xmax - xmin) / (W - 1))
    print >>msgout, u"   %.5f ≤ y ≤ %.5f, H = %d, ∆y = %.5f" % \
        (ymin, ymax, H, (ymax - ymin) / (H - 1))
    print >>msgout, "Data type: %d-bit %s" % (
        8 * data.dtype.alignment, kind_str[data.dtype.kind])
    print >>msgout, u"Scattering angle: θ = %.5f (%.2f°)" % (
        theta, theta * 180 / pi)
    print >>msgout, ""

    if args.alpha is None:
        args.alpha = 3 * \
            sqrt((xmax - xmin) * (ymax - ymin) / (H - 1) / (W - 1))
    else:
        try:
            args.alpha = float(args.alpha)
        except ValueError:
            print >>sys.stderr, "Bad parameter for alpha. Expected floating point or integer, got '%s' instead." % args.alpha
            sys.exit()

    brt_inv = FMSBrokenRayInversion(H, W, xmin, xmax, ymin, ymax, theta)

    print >>msgout, u"Performing inversion with α = %.5f." % args.alpha

    print >>msgout, "Generating Kernel...",
    msgout.flush()
    t0 = time.time()
    kernel = brt_inv.make_kernel(args.alpha)
    print >>msgout, "%.2f seconds" % (time.time() - t0)

    print >>msgout, "Performing inversion...",
    msgout.flush()
    t0 = time.time()
    reconstructed = brt_inv.reconstruct(data, kernel, args.alpha)
    print >>msgout, "%.2f seconds" % (time.time() - t0)

    output_metadata = array((xmin, xmax, ymin, ymax))
elif brt_type == "QBRT":
    from polar import PolarBrokenRayInversion
    import parallel
    try:
        smin, smax, sigmamin, sigmamax, phi = metadata
    except ValueError:
        print >>sys.stderr, "Unexpected end of data!"
        raise
    print >>msgout, "Inversion of the Polar Broken Ray transform."
    print >>msgout, u"Array information:"
    print >>msgout, u"   %.5f ≤ s ≤ %.5f, W = %d, ∆s = %.5f" % \
        (smin, smax, W, (smax - smin) / (W - 1))
    print >>msgout, u"   %.5f ≤ σ ≤ %.5f, H = %d, ∆σ = %.5f" % \
        (sigmamin, sigmamax, H, (sigmamax - sigmamin) / (H - 1))
    print >>msgout, "Data type: %d-bit %s" % (
        8 * data.dtype.alignment, kind_str[data.dtype.kind])
    print >>msgout, u"Scattering angle: ϕ = %.5f (%.2f°)" % (
        phi, phi * 180 / pi)
    print >>msgout, ""

    if args.alpha is None:
        args.alpha = 2 * (smax - smin) / (W - 1)
    else:
        try:
            args.alpha = float(args.alpha)
        except ValueError:
            print >>sys.stderr, "Bad parameter for alpha. Expected floating point or integer, got '%s' instead." % args.alpha
            sys.exit()

    if args.threads is not None:
        try:
            args.threads = int(args.threads)
        except ValueError:
            print >>sys.stderr, "Bad parameter for threads. Expected integer, got '%s' instead." % args.threads
            sys.exit()

    if args.nmax is None:
        args.nmax = H / 4
    else:
        try:
            args.nmax = int(args.nmax)
        except ValueError:
            print >>sys.stderr, "Bad parameter for nmax. Expected integer, got '%s' instead." % args.nmax
            sys.exit()

    if args.threads > 0:
        print >>msgout, u"Performing inversion with α = %.5f, nmax = %d, threads = %d." % (
                        args.alpha, args.nmax, args.threads)
    else:
        print >>msgout, u"Performing inversion with α = %.5f, nmax = %d, threads = autodetect." % (
                        args.alpha, args.nmax)

    qinv = PolarBrokenRayInversion(
        data, phi, smin, smax, args.alpha, nmax=args.nmax)
    jm = parallel.JobManager(threads=args.threads)
    jm.jobqueue.put(qinv)
    while True:
        with qinv.status:
            qinv.status.wait()
            print >>msgout, "Performing inversion: %.2f%% done.\r" % (
                qinv.jobsdone * 100.0 / qinv.jobcount),
            msgout.flush()
            if qinv.jobsdone >= qinv.jobcount or not qinv.running:
                break
    print >>msgout, ""
    reconstructed = qinv.reconstruct().real
    output_metadata = array(
        (qinv.rmin, qinv.rmax, qinv.thetamin, qinv.thetamax))
else:
    print >>sys.stderr, "Error: Unknown Broken Ray transform type: '%s'" % brt_type
    sys.exit()

print >>msgout, "Writing data to '%s'..." % args.output,
msgout.flush()
t0 = time.time()
if args.output != "-":
    outfile = open(args.output, "wb")
npsave(outfile, "data")
npsave(outfile, output_metadata)
npsave(outfile, reconstructed)
print >>msgout, "%.2f seconds" % (time.time() - t0)
if outfile is not sys.stdin:
    outfile.close()
