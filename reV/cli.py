# -*- coding: utf-8 -*-
"""
reV command line interface (CLI).
"""
import click

from reV.batch.cli_batch import from_config as run_batch_from_config
from reV.handlers.cli_collect import from_config as run_collect_from_config
from reV.handlers.cli_multi_year import from_config as run_my_from_config
from reV.econ.cli_econ import from_config as run_econ_from_config
from reV.generation.cli_gen import from_config as run_gen_from_config
from reV.offshore.cli_offshore import from_config as run_offshore_from_config
from reV.pipeline.cli_pipeline import from_config as run_pipeline_from_config
from reV.rep_profiles.cli_rep_profiles import from_config as run_rp_from_config
from reV.supply_curve.cli_sc_aggregation import (from_config
                                                 as run_sc_agg_from_config)
from reV.supply_curve.cli_supply_curve import from_config as run_sc_from_config
from reV.qa_qc.cli_qa_qc import from_config as run_qa_qc_from_config

from rex.utilities.cli_dtypes import STR


@click.group()
@click.option('--name', '-n', default='reV', type=STR,
              help='Job name. Default is "reV".')
@click.option('--config_file', '-c',
              required=True, type=click.Path(exists=True),
              help='reV configuration file json for a single module.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, config_file, verbose):
    """reV command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def generation(ctx, verbose):
    """Generation analysis (pv, csp, windpower, etc...)."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_gen_from_config, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def econ(ctx, verbose):
    """Econ analysis (lcoe, single-owner, etc...)."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_econ_from_config, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def offshore(ctx, verbose):
    """Offshore gen/econ aggregation with ORCA."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_offshore_from_config, config_file=config_file,
               verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def collect(ctx, verbose):
    """Collect files from a job run on multiple nodes."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_collect_from_config, config_file=config_file,
               verbose=verbose)


@main.command()
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def pipeline(ctx, cancel, monitor, verbose):
    """Execute multiple steps in a reV analysis pipeline."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_pipeline_from_config, config_file=config_file,
               cancel=cancel, monitor=monitor, verbose=verbose)


@main.command()
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given batch.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def batch(ctx, dry_run, cancel, verbose):
    """Execute multiple steps in a reV analysis pipeline."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_batch_from_config, config_file=config_file,
               dry_run=dry_run, cancel=cancel, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def multi_year(ctx, verbose):
    """Run reV multi_year using the config file."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_my_from_config, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def supply_curve_aggregation(ctx, verbose):
    """Run reV supply curve aggregation using the config file."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_sc_agg_from_config, config_file=config_file,
               verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def supply_curve(ctx, verbose):
    """Run reV supply curve using the config file."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_sc_from_config, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def rep_profiles(ctx, verbose):
    """Run reV representative profiles using the config file."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_rp_from_config, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def qa_qc(ctx, verbose):
    """Run reV qa-qc using the config file."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(run_qa_qc_from_config, config_file=config_file, verbose=verbose)


if __name__ == '__main__':
    main(obj={})
