"""
Batch Job CLI entry points.
"""
import click
from reV.generation.cli_gen import main
from reV.batch.batch import BatchJob


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV batch configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV batch from a config file."""
    verbose = any([verbose, ctx.obj['VERBOSE']])
    BatchJob.run(config_file)


if __name__ == '__main__':
    main(obj={})
