# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Pipeline CLI entry points.
"""
import click

from reV.config.pipeline import PipelineConfig
from reV.pipeline.pipeline import Pipeline
from reV import __version__

from rex.utilities.cli_dtypes import STR
from rex.utilities.utilities import get_class_properties
from rex.utilities.execution import SubprocessManager


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default='reV-pipeline', type=STR,
              show_default=True,
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
def valid_config_keys():
    """
    Echo the valid Pipeline config keys
    """
    click.echo(', '.join(get_class_properties(PipelineConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV pipeline configuration json file.')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('--background', is_flag=True,
              help='Flag to monitor pipeline jobs continuously in the '
              'background using the nohup command. This only works with the '
              '--monitor flag. Note that the stdout/stderr will not be '
              'captured, but you can set a pipeline log_file to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, cancel, monitor, background, verbose):
    """Run reV pipeline from a config file."""
    verbose = any([verbose, ctx.obj['VERBOSE']])

    if cancel:
        Pipeline.cancel_all(config_file)
    elif monitor and background:
        pipeline_monitor_background(config_file, verbose=verbose)
    else:
        Pipeline.run(config_file, monitor=monitor, verbose=verbose)


def pipeline_monitor_background(config_file, verbose=False):
    """Submit the pipeline execution with monitoring in the background using
    the nohup linux command.

    Parameters
    ----------
    config_file : STR
        Filepath to reV pipeline configuration json file.
    verbose : bool
        Flag to turn on debug logging for the pipeline.
    """

    cmd = ('python -m reV.cli -c {} pipeline --monitor'.format(config_file))
    if verbose:
        cmd += ' -v'

    SubprocessManager.submit(cmd, background=True, background_stdout=False)


if __name__ == '__main__':
    main(obj={})
