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
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.pass_context
def from_config(ctx, config_file, verbose, dry_run):
    """Run reV batch from a config file."""
    verbose = any([verbose, ctx.obj['VERBOSE']])
    BatchJob.run(config_file, dry_run=dry_run)


if __name__ == '__main__':
    main(obj={})
