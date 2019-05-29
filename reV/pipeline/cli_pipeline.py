"""
Generation CLI entry points.
"""
import click
from reV.pipeline.pipeline import Pipeline


@click.group()
@click.option('--name', '-n', default='reV', type=str,
              help='reV job name. Default is "reV".')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """Command line interface (CLI) for the reV 2.0 Full Pipeline."""
    ctx.obj['NAME'] = name
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
    verbose = any([verbose, ctx.obj['VERBOSE']])

    # init pipeline config, which will also run pre-flight checks
    Pipeline.run(config_file)


if __name__ == '__main__':
    main(obj={})
