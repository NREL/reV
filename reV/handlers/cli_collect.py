"""
Generation CLI entry points.
"""
import click
import logging
import os
import pprint
import time

from reV.config.collection import CollectionConfig
from reV.handlers.collection import Collector
from reV.utilities.cli_dtypes import STR, STRLIST
from reV.utilities.loggers import init_mult


logger = logging.getLogger(__name__)


@click.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV generation configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV gen from a config file."""
    name = ctx.obj['NAME']

    # Instantiate the config object
    config = CollectionConfig(config_file)

    # take name from config if not default
    if config.name.lower() != 'rev':
        name = config.name
        ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.logging_level == logging.DEBUG:
        verbose = True

    # make output directory if does not exist
    if not os.path.exists(config.dirout):
        os.makedirs(config.dirout)

    # initialize loggers.
    init_mult(name, config.logdir, modules=[__name__, 'reV.handlers'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV 2.0 collection from config file: "{}"'
                .format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.logdir))
    logger.info('Target collection directory: "{}"'.format(config.coldir))
    logger.info('The following project points were specified: "{}"'
                .format(config.get('project_points', None)))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    # set config objects to be passed through invoke to direct methods
    ctx.obj['H5_DIR'] = config.coldir
    ctx.obj['DSETS'] = config.dsets
    ctx.obj['PROJECT_POINTS'] = config.project_points
    ctx.obj['PARALLEL'] = config.parallel
    ctx.obj['VERBOSE'] = verbose

    for file_prefix in config.file_prefixes:
        ctx.obj['H5_FILE'] = os.path.join(config.dirout, file_prefix + '.h5')
        ctx.obj['FILE_PREFIX'] = file_prefix

        if config.execution_control.option == 'local':
            ctx.invoke(collect)


@click.group()
@click.option('--h5_file', '-f', required=True, type=str,
              help='H5 file to be collected into.')
@click.option('--h5_dir', '-d', required=True, type=click.Path(exists=True),
              help='Directory containing h5 files to collect.')
@click.option('--project_points', '-pp', required=True,
              type=click.File(exists=True),
              help='Project points file representing the full '
              'collection scope.')
@click.option('--dsets', '-ds', required=True, type=STRLIST,
              help='Dataset names to be collected.')
@click.option('--file_prefix', '-fp', type=STR, default=None,
              help='File prefix found in the h5 file names to be collected.')
@click.option('-par', '--parallel', is_flag=True,
              help='Flag to turn on parallel collection.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def main(ctx, h5_file, h5_dir, project_points, dsets, file_prefix, parallel,
         verbose):
    """Main entry point for collection."""

    ctx.obj['H5_FILE'] = h5_file
    ctx.obj['H5_DIR'] = h5_dir
    ctx.obj['PROJECT_POINTS'] = project_points
    ctx.obj['DSETS'] = dsets
    ctx.obj['FILE_PREFIX'] = file_prefix
    ctx.obj['PARALLEL'] = parallel
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def collect(ctx, verbose):
    """Run collection on local worker."""

    name = ctx.obj['NAME']
    h5_file = ctx.obj['H5_FILE']
    h5_dir = ctx.obj['H5_DIR']
    project_points = ctx.obj['PROJECT_POINTS']
    dsets = ctx.obj['DSETS']
    file_prefix = ctx.obj['FILE_PREFIX']
    parallel = ctx.obj['PARALLEL']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # initialize loggers for multiple modules
    init_mult(name, h5_dir, modules=[__name__, 'reV.handlers'],
              verbose=verbose, node=True)

    for key, val in ctx.obj.items():
        logger.debug('ctx var passed to collection method: "{}" : "{}" '
                     'with type "{}"'.format(key, val, type(val)))

    logger.info('Collection is being run for "{}" with with job name "{}" '
                'and collection dir: {}. Target output path is: {}'
                .format(dsets, name, h5_dir, h5_file))
    t0 = time.time()

    Collector.collect(h5_file, h5_dir, project_points, dsets[0],
                      file_prefix=file_prefix, parallel=parallel)
    if len(dsets) > 1:
        for dset_name in dsets[1:]:
            Collector.add_dataset(h5_file, h5_dir, dset_name,
                                  file_prefix=file_prefix, parallel=parallel)

    logger.info('Collection complete from h5 directory: "{0}". '
                'Time elapsed: {1:.2f} min. Target output file: "{2}"'
                .format(h5_dir, (time.time() - t0) / 60, h5_file))


if __name__ == '__main__':
    main(obj={})
