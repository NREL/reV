# -*- coding: utf-8 -*-
"""
reV offshore wind farm aggregation module command line interface (CLI).

This module aggregates offshore data from high res wind resource data to
coarse wind farm sites and then calculates the ORCA econ data.

Offshore resource / generation data refers to WTK 2km (fine resolution)
Offshore farms refer to ORCA data on 600MW wind farms (coarse resolution)
"""

import os
import click
import logging
import time

from reV.utilities.cli_dtypes import STR, INT, PROJECTPOINTS, SAMFILES
from reV.utilities.loggers import init_mult
from reV.pipeline.status import Status
from reV.offshore.offshore import Offshore


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--name', '-n', default='off', type=STR,
              help='Job name. Default is "off".')
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV wind generation/econ output file.')
@click.option('--offshore_fpath', '-of', type=STR, required=True,
              help='reV wind farm meta and ORCA cost data inputs.')
@click.option('--points', '-p', default=slice(0, 100), type=PROJECTPOINTS,
              help=('reV project points to analyze '
                    '(slice, list, or file string). '
                    'Default is slice(0, 100)'))
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files (required) (str, dict, or list).')
@click.option('--max_workers', '-mw', type=INT, default=None,
              help='Max workers to use. None is all workers, 1 is serial.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save offshore logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, gen_fpath, offshore_fpath, points, sam_files,
         log_dir, verbose):
    """Main entry point to run offshore wind aggregation"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['OFFSHORE_FPATH'] = offshore_fpath
    ctx.obj['POINTS'] = points
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV.offshore'],
                  verbose=verbose, node=True)

        fpath_out = gen_fpath.replace('.h5', '_offshore.h5')

        try:
            Offshore.run(gen_fpath, offshore_fpath, points, sam_files,
                         fpath_out=fpath_out, max_workers=None)
        except Exception as e:
            logger.exception('Offshore module failed, received the '
                             'following exception:\n{}'.format(e))
            raise e

        runtime = (time.time() - t0) / 60

        status = {'dirout': os.path.dirname(fpath_out),
                  'fout': os.path.basename(fpath_out),
                  'job_status': 'successful',
                  'runtime': runtime, 'finput': gen_fpath}
        Status.make_job_file(os.path.dirname(fpath_out), 'offshore',
                             name, status)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV offshore CLI.')
        raise
