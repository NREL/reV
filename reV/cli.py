# -*- coding: utf-8 -*-
"""
reV command line interface (CLI).
"""
import click
import logging

from reV.batch.cli_batch import from_config as run_batch_from_config
from reV.batch.cli_batch import valid_config_keys as batch_keys
from reV.handlers.cli_collect import from_config as run_collect_from_config
from reV.handlers.cli_collect import valid_config_keys as collect_keys
from reV.handlers.cli_multi_year import from_config as run_my_from_config
from reV.handlers.cli_multi_year import valid_config_keys as my_keys
from reV.econ.cli_econ import from_config as run_econ_from_config
from reV.econ.cli_econ import valid_config_keys as econ_keys
from reV.generation.cli_gen import from_config as run_gen_from_config
from reV.generation.cli_gen import valid_config_keys as gen_keys
from reV.offshore.cli_offshore import from_config as run_offshore_from_config
from reV.offshore.cli_offshore import valid_config_keys as offshore_keys
from reV.pipeline.cli_pipeline import from_config as run_pipeline_from_config
from reV.pipeline.cli_pipeline import valid_config_keys as pipeline_keys
from reV.rep_profiles.cli_rep_profiles import from_config as run_rp_from_config
from reV.rep_profiles.cli_rep_profiles import (valid_config_keys
                                               as rep_profiles_keys)
from reV.supply_curve.cli_sc_aggregation import (from_config
                                                 as run_sc_agg_from_config)
from reV.supply_curve.cli_sc_aggregation import (valid_config_keys
                                                 as sc_agg_keys)
from reV.supply_curve.cli_supply_curve import from_config as run_sc_from_config
from reV.supply_curve.cli_supply_curve import valid_config_keys as sc_keys
from reV.qa_qc.cli_qa_qc import from_config as run_qa_qc_from_config
from reV.qa_qc.cli_qa_qc import valid_config_keys as qa_qc_keys
from reV import __version__

from rex.utilities.cli_dtypes import STR

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
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


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def generation(ctx, verbose):
    """Generation analysis (pv, csp, windpower, etc...)."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_gen_from_config, config_file=config_file,
                   verbose=verbose)


@generation.command()
@click.pass_context
def valid_generation_keys(ctx):
    """
    Valid Generation config keys
    """
    ctx.invoke(gen_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def econ(ctx, verbose):
    """Econ analysis (lcoe, single-owner, etc...)."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_econ_from_config, config_file=config_file,
                   verbose=verbose)


@econ.command()
@click.pass_context
def valid_econ_keys(ctx):
    """
    Valid Econ config keys
    """
    ctx.invoke(econ_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def offshore(ctx, verbose):
    """Offshore gen/econ aggregation with NRWAL."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_offshore_from_config, config_file=config_file,
                   verbose=verbose)


@offshore.command()
@click.pass_context
def valid_offshore_keys(ctx):
    """
    Valid offshore config keys
    """
    ctx.invoke(offshore_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def collect(ctx, verbose):
    """Collect files from a job run on multiple nodes."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_collect_from_config, config_file=config_file,
                   verbose=verbose)


@collect.command()
@click.pass_context
def valid_collect_keys(ctx):
    """
    Valid Collect config keys
    """
    ctx.invoke(collect_keys)


@main.group(invoke_without_command=True)
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('--background', is_flag=True,
              help='Flag to monitor pipeline jobs continuously '
              'in the background using the nohup command. Note that the '
              'stdout/stderr will not be captured, but you can set a '
              'pipeline "log_file" to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def pipeline(ctx, cancel, monitor, background, verbose):
    """Execute multiple steps in a reV analysis pipeline."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_pipeline_from_config, config_file=config_file,
                   cancel=cancel, monitor=monitor, background=background,
                   verbose=verbose)


@pipeline.command()
@click.pass_context
def valid_pipeline_keys(ctx):
    """
    Valid Pipeline config keys
    """
    ctx.invoke(pipeline_keys)


@main.group(invoke_without_command=True)
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given batch.')
@click.option('--delete', is_flag=True,
              help='Flag to delete all batch job sub directories associated '
              'with the batch_jobs.csv in the current batch config directory.')
@click.option('--monitor-background', is_flag=True,
              help='Flag to monitor all batch pipelines continuously '
              'in the background using the nohup command. Note that the '
              'stdout/stderr will not be captured, but you can set a '
              'pipeline "log_file" to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def batch(ctx, dry_run, cancel, delete, monitor_background, verbose):
    """Execute multiple steps in a reV analysis pipeline."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_batch_from_config, config_file=config_file,
                   dry_run=dry_run, cancel=cancel, delete=delete,
                   monitor_background=monitor_background,
                   verbose=verbose)


@batch.command()
@click.pass_context
def valid_batch_keys(ctx):
    """
    Valid Batch config keys
    """
    ctx.invoke(batch_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def multi_year(ctx, verbose):
    """Run reV multi year using the config file."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_my_from_config, config_file=config_file,
                   verbose=verbose)


@multi_year.command()
@click.pass_context
def valid_multi_year_keys(ctx):
    """
    Valid Multi Year config keys
    """
    ctx.invoke(my_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def supply_curve_aggregation(ctx, verbose):
    """Run reV supply curve aggregation using the config file."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_sc_agg_from_config, config_file=config_file,
                   verbose=verbose)


@supply_curve_aggregation.command()
@click.pass_context
def valid_supply_curve_aggregation_keys(ctx):
    """
    Valid Supply Curve Aggregation config keys
    """
    ctx.invoke(sc_agg_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def supply_curve(ctx, verbose):
    """Run reV supply curve using the config file."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_sc_from_config, config_file=config_file,
                   verbose=verbose)


@supply_curve.command()
@click.pass_context
def valid_supply_curve_keys(ctx):
    """
    Valid Supply Curve config keys
    """
    ctx.invoke(sc_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def rep_profiles(ctx, verbose):
    """Run reV representative profiles using the config file."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_rp_from_config, config_file=config_file,
                   verbose=verbose)


@rep_profiles.command()
@click.pass_context
def valid_rep_profiles_keys(ctx):
    """
    Valid Representative Profiles config keys
    """
    ctx.invoke(rep_profiles_keys)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def qa_qc(ctx, verbose):
    """Run reV QA/QC using the config file."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(run_qa_qc_from_config, config_file=config_file,
                   verbose=verbose)


@qa_qc.command()
@click.pass_context
def valid_qa_qc_keys(ctx):
    """
    Valid QA/QC config keys
    """
    ctx.invoke(qa_qc_keys)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV CLI')
        raise
