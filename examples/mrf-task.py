import os
import sys
import copy
import argparse


def main(argv=sys.argv[1:]):
    # Parse the input
    parser = argparse.ArgumentParser(description='This script implement MRF on low-resolution image with the help of high-resolution images.')
    parser.add_argument('dir_lowres', help='string, directory of input low-resolution image.')
    parser.add_argument('dir_hires_b', help='string, directory of input high-resolution blue-band image (typically g-band).')
    parser.add_argument('dir_hires_r', help='string, directory of input high-resolution red-band image (typically r-band).')
    parser.add_argument('config', help='configuration file, in the format of "yaml".')
    parser.add_argument('--galcat', help='string, directory of a catalog (in ascii format) which contains RA and DEC of galaxies which you want to retain during MRF.')
    parser.add_argument('--output', help='The prefix of output files. Default is "mrf".')
    parser.add_argument('--shutup', action="store_true", help='Whether print out the process.')
    args = parser.parse_args()

    from mrf.task import MrfTask
    task = MrfTask(args.config)

    if args.output:
        output_name = args.output
    else:
        output_name = 'mrf'

    if args.shutup:
        verb = args.verbose
    else:
        verb = True
    
    results = task.run(args.dir_lowres, args.dir_hires_b, args.dir_hires_r, args.galcat, output_name=output_name, verbose=verb)

if __name__ == "__main__":
    main()