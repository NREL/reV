"""
Generation CLI entry points.
"""
import click
import json
from reV.utilities.cli_dtypes import STR
from reV.utilities.loggers import init_logger
from reV.pipeline.pipeline import Pipeline


@click.group()
@click.option('--name', '-n', default='reV', type=str,
              help='reV job name. Default is "reV".')
@click.option('--status_dir', '-st', default=None, type=STR,
              help='Optional directory containing reV status json.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, status_dir, verbose):
    """Command line interface (CLI) for the reV 2.0 Full Pipeline."""
    ctx.obj['NAME'] = name
    ctx.obj['STATUS_DIR'] = status_dir
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV pipeline configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV pipeline from a config file."""
    status_dir = ctx.obj['STATUS_DIR']
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # load the config file as a dict
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    msg = ('The reV pipeline config must have a "pipeline" entry which is a '
           'list of command/config key pairs.')
    if 'pipeline' not in config_dict:
        raise KeyError(msg)
    if not isinstance(config_dict['pipeline'], list):
        raise TypeError(msg)

    if 'logging' in config_dict:
        init_logger('reV.pipeline', **config_dict['logging'])

    Pipeline.run(config_dict['pipeline'], status_dir)


if __name__ == '__main__':
    main(obj={})
