# -*- coding: utf-8 -*-
"""
Pipeline CLI entry points.
"""
import click

from reV.pipeline.pipeline import Pipeline
from rex.utilities.cli_dtypes import STR


@click.group()
@click.option('--name', '-n', default='reV-pipeline', type=STR,
              help='reV pipeline name, by default "reV-pipeline".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Pipeline Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV pipeline configuration json file.')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, cancel, monitor, verbose):
    """Run reV pipeline from a config file."""
    verbose = any([verbose, ctx.obj['VERBOSE']])

    if cancel:
        Pipeline.cancel_all(config_file)
    else:
        Pipeline.run(config_file, monitor=monitor, verbose=verbose)


if __name__ == '__main__':
    main(obj={})
