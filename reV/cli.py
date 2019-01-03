"""
Generation
"""
import click
import logging

from reV.cli_dtypes import STR
from reV.generation.cli_gen import from_config as run_gen_from_config


logger = logging.getLogger(__name__)


@click.group()
@click.option('--name', '-n', default='reV', type=STR,
              help='Job name. Default is "reV".')
@click.option('--config_file', '-c', type=STR,
              help='reV configuration file json for a single module.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, config_file, verbose):
    """reV 2.0 config command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['name'] = name
    ctx.obj['config_file'] = config_file
    ctx.obj['verbose'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def generation(ctx, verbose):
    """Run reV 2.0 generation using the config file."""
    config_file = ctx.obj['config_file']
    verbose = any([verbose, ctx.obj['verbose']])
    ctx.invoke(run_gen_from_config, config_file=config_file, verbose=verbose)


if __name__ == '__main__':
    main(obj={})
