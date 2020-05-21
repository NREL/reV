# -*- coding: utf-8 -*-
"""
Batch Job CLI entry points.
"""
import click

from reV.batch.batch import BatchJob
from rex.utilities.cli_dtypes import STR


@click.group()
@click.option('--name', '-n', default='reV-batch', type=STR,
              help='reV batch name, by default "reV-batch".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Batch Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NAME'] = name


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV batch configuration json file.')
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, dry_run, cancel, verbose):
    """Run reV batch from a config file."""
    verbose = any([verbose, ctx.obj['VERBOSE']])

    if cancel:
        BatchJob.cancel_all(config_file)
    else:
        BatchJob.run(config_file, dry_run=dry_run)


if __name__ == '__main__':
    main(obj={})
