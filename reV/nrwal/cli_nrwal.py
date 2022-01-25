# -*- coding: utf-8 -*-
# pylint: disable=all
"""
reV-NRWAL module command line interface (CLI).

This module runs reV data through the NRWAL compute library. This code was
first developed to use a custom offshore wind LCOE equation library but has
since been refactored to analyze any equation library in NRWAL.

Everything in this module operates on the spatiotemporal resolution of the reV
generation output file. This is usually the wind or solar resource resolution
but could be the supply curve resolution after representative profiles is run.
"""
import pprint
import os
import click
import logging
import time

from reV.config.nrwal_config import RevNrwalConfig
from reV.pipeline.status import Status
from reV.nrwal.nrwal import RevNrwal
from reV.utilities.cli_dtypes import SAMFILES
from reV.utilities import ModuleName
from reV import __version__

from rex.utilities.cli_dtypes import STR, INT, STRLIST
from rex.utilities.loggers import init_mult
from rex.utilities.hpc import SLURM
from rex.utilities.utilities import get_class_properties

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default=os.path.basename(os.getcwd()),
              type=STR, show_default=True, help='reV NRWAL job name.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV-NRWAL Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
def valid_config_keys():
    """
    Echo the valid nrwal config keys
    """
    click.echo(', '.join(get_class_properties(RevNrwalConfig)))


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='reV-NRWAL configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run reV-NRWAL analysis from a config file."""
    # Instantiate the config object
    config = RevNrwalConfig(config_file)

    # take name from config
    name = ctx.obj['NAME'] = config.name

    # Enforce verbosity if logging level is specified in the config
    if config.log_level == logging.DEBUG:
        verbose = True

    # initialize loggers
    init_mult(name, config.log_directory, modules=[__name__, 'reV', 'rex'],
              verbose=verbose)

    # Initial log statements
    logger.info('Running reV-NRWAL analysis from config '
                'file: "{}"'.format(config_file))
    logger.info('Target output directory: "{}"'.format(config.dirout))
    logger.info('Target logging directory: "{}"'.format(config.log_directory))
    logger.debug('The full configuration input is as follows:\n{}'
                 .format(pprint.pformat(config, indent=4)))

    gen_fpaths = config.gen_fpath
    if isinstance(gen_fpaths, str):
        gen_fpaths = [gen_fpaths]

    for i, gen_fpath in enumerate(gen_fpaths):
        job_name = '{}_{}'.format(name, str(i).zfill(2))
        ctx.obj['NAME'] = job_name

        if config.execution_control.option == 'local':
            status = Status.retrieve_job_status(config.dirout,
                                                module=ModuleName.NRWAL,
                                                job_name=job_name)
            if status != 'successful':
                Status.add_job(
                    config.dirout, module=ModuleName.NRWAL,
                    job_name=job_name, replace=True,
                    job_attrs={'hardware': 'local',
                               'fout': '{}_nrwal.h5'.format(job_name),
                               'dirout': config.dirout,
                               'finput': gen_fpath})
                ctx.invoke(direct,
                           gen_fpath=gen_fpath,
                           site_data=config.site_data,
                           sam_files=config.sam_files,
                           nrwal_configs=config.nrwal_configs,
                           output_request=config.output_request,
                           save_raw=config.save_raw,
                           meta_gid_col=config.meta_gid_col,
                           site_meta_cols=config.site_meta_cols,
                           log_dir=config.log_directory,
                           verbose=verbose)

        elif config.execution_control.option in ('eagle', 'slurm'):
            ctx.obj['GEN_FPATH'] = gen_fpath
            ctx.obj['SITE_DATA'] = config.site_data
            ctx.obj['SAM_FILES'] = config.sam_files
            ctx.obj['NRWAL_CONFIGS'] = config.nrwal_configs
            ctx.obj['OUTPUT_REQUEST'] = config.output_request
            ctx.obj['SAVE_RAW'] = config.save_raw
            ctx.obj['META_GID_COL'] = config.meta_gid_col
            ctx.obj['SITE_META_COLS'] = config.site_meta_cols
            ctx.obj['OUT_DIR'] = config.dirout
            ctx.obj['LOG_DIR'] = config.log_directory
            ctx.obj['VERBOSE'] = verbose

            ctx.invoke(slurm,
                       alloc=config.execution_control.allocation,
                       memory=config.execution_control.memory,
                       walltime=config.execution_control.walltime,
                       feature=config.execution_control.feature,
                       module=config.execution_control.module,
                       conda_env=config.execution_control.conda_env,
                       sh_script=config.execution_control.sh_script)


@main.group(invoke_without_command=True)
@click.option('--gen_fpath', '-gf', type=STR, required=True,
              help='reV wind generation/econ output file. Anything in the '
              'output_request is added and/or manipulated in this file.')
@click.option('--site_data', '-sd', type=STR, required=True,
              help='Site-specific input data for NRWAL calculation. String '
              'should be a filepath that points to a csv. Rows match sites, '
              'columns are input keys. Need a "gid" column that corresponds '
              'to the "meta_gid_col" in the gen_fpath meta data and a '
              '"config" column that corresponds to the nrwal_configs input. '
              'Only sites with a gid in this files "gid" column will be run '
              'through NRWAL.')
@click.option('--sam_files', '-sf', required=True, type=SAMFILES,
              help='SAM config files lookup mapping config keys to config '
              'filepaths. (required) (dict). Should have the same config '
              'keys as the nrwal_configs input.')
@click.option('--nrwal_configs', '-nc', required=True, type=SAMFILES,
              help='NRWAL config files lookup mapping config keys to config '
              'filepaths. (required) (dict). Should have the same config '
              'keys as the sam_files input.')
@click.option('--output_request', '-or', default=None, type=STRLIST,
              help='List of output dataset names you want written to the '
              'gen_fpath file. Any key from the NRWAL configs or any of the '
              'inputs (site_data or sam_files) is available to be exported as '
              'an output dataset. If you want to manipulate a dset like '
              'cf_mean from gen_fpath and include it in the output_request, '
              'you should set save_raw=True and then in the NRWAL equations '
              'use cf_mean_raw as the input and then define cf_mean as the '
              'manipulated data that will be included in the output_request.')
@click.option('--save_raw', '-sr', default=True, type=bool,
              help='Flag to save a copy of existing datasets in gen_fpath '
              'that are part of the output_request. For example, if you '
              'request cf_mean in output_request and manipulate the cf_mean '
              'dataset in the NRWAL equations, the original cf_mean will be '
              'archived under the "cf_mean_raw" dataset in gen_fpath.')
@click.option('--meta_gid_col', '-mg', default='gid', type=str,
              help='Column label in the source meta data from gen_fpath that '
              'contains the unique gid identifier. This will be joined to the '
              'site_data "gid" column.')
@click.option('--site_meta_cols', '-mc', default=None, type=STRLIST,
              help='Column labels from site_data to pass through to the '
              'output meta data. None (default) will use class variable '
              'DEFAULT_META_COLS, and any additional cols requested here will '
              'be added to DEFAULT_META_COLS.')
@click.option('--log_dir', '-ld', type=STR, default='./logs/',
              help='Directory to save reV-NRWAL logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, gen_fpath, site_data, sam_files, nrwal_configs,
           output_request, save_raw, meta_gid_col,
           site_meta_cols, log_dir, verbose):
    """Main entry point to run reV-NRWAL analysis"""
    name = ctx.obj['NAME']
    ctx.obj['GEN_FPATH'] = gen_fpath
    ctx.obj['SITE_DATA'] = site_data
    ctx.obj['SAM_FILES'] = sam_files
    ctx.obj['NRWAL_CONFIGS'] = nrwal_configs
    ctx.obj['OUTPUT_REQUEST'] = output_request
    ctx.obj['SAVE_RAW'] = save_raw
    ctx.obj['META_GID_COL'] = meta_gid_col
    ctx.obj['SITE_META_COLS'] = site_meta_cols
    ctx.obj['LOG_DIR'] = log_dir
    ctx.obj['VERBOSE'] = verbose

    if ctx.invoked_subcommand is None:
        t0 = time.time()
        init_mult(name, log_dir, modules=[__name__, 'reV', 'rex'],
                  verbose=verbose, node=True)

        try:
            RevNrwal.run(gen_fpath, site_data, sam_files, nrwal_configs,
                         output_request,
                         save_raw=save_raw,
                         meta_gid_col=meta_gid_col,
                         site_meta_cols=site_meta_cols)
        except Exception as e:
            logger.exception('reV-NRWAL module failed, received the '
                             'following exception:\n{}'.format(e))
            raise e

        runtime = (time.time() - t0) / 60

        status = {'dirout': os.path.dirname(gen_fpath),
                  'fout': os.path.basename(gen_fpath),
                  'job_status': 'successful',
                  'runtime': runtime, 'finput': gen_fpath}
        Status.make_job_file(os.path.dirname(gen_fpath), ModuleName.NRWAL,
                             name, status)


def get_node_cmd(name, gen_fpath, site_data, sam_files, nrwal_configs,
                 output_request, save_raw, meta_gid_col, site_meta_cols,
                 log_dir, verbose):
    """Get a CLI call command for the reV-NRWAL cli."""

    args = ['-gf {}'.format(SLURM.s(gen_fpath)),
            '-sd {}'.format(SLURM.s(site_data)),
            '-sf {}'.format(SLURM.s(sam_files)),
            '-nc {}'.format(SLURM.s(nrwal_configs)),
            '-or {}'.format(SLURM.s(output_request)),
            '-sr {}'.format(SLURM.s(save_raw)),
            '-mg {}'.format(SLURM.s(meta_gid_col)),
            '-mc {}'.format(SLURM.s(site_meta_cols)),
            '-ld {}'.format(SLURM.s(log_dir)),
            ]

    if verbose:
        args.append('-v')

    cmd = ('python -m reV.nrwal.cli_nrwal -n {} direct {}'
           .format(SLURM.s(name), ' '.join(args)))
    logger.debug('Creating the following command line call:\n\t{}'.format(cmd))

    return cmd


@direct.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='SLURM allocation account name.')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--memory', '-mem', default=None, type=INT, help='SLURM node '
              'memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='SLURM walltime request in hours. Default is 1.0')
@click.option('--module', '-mod', default=None, type=STR,
              help='Module to load')
@click.option('--conda_env', '-env', default=None, type=STR,
              help='Conda env to activate')
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.option('--sh_script', '-sh', default=None, type=STR,
              show_default=True,
              help='Extra shell script commands to run before the reV call.')
@click.pass_context
def slurm(ctx, alloc, feature, memory, walltime, module, conda_env,
          stdout_path, sh_script):
    """slurm (Eagle) submission tool for reV supply curve aggregation."""

    name = ctx.obj['NAME']
    gen_fpath = ctx.obj['GEN_FPATH']
    site_data = ctx.obj['SITE_DATA']
    sam_files = ctx.obj['SAM_FILES']
    nrwal_configs = ctx.obj['NRWAL_CONFIGS']
    output_request = ctx.obj['OUTPUT_REQUEST']
    save_raw = ctx.obj['SAVE_RAW']
    meta_gid_col = ctx.obj['META_GID_COL']
    site_meta_cols = ctx.obj['SITE_META_COLS']
    out_dir = ctx.obj['OUT_DIR']
    log_dir = ctx.obj['LOG_DIR']
    verbose = ctx.obj['VERBOSE']

    if stdout_path is None:
        stdout_path = os.path.join(log_dir, 'stdout/')

    cmd = get_node_cmd(name, gen_fpath, site_data, sam_files, nrwal_configs,
                       output_request, save_raw, meta_gid_col, site_meta_cols,
                       log_dir, verbose)
    if sh_script:
        cmd = sh_script + '\n' + cmd

    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir, module=ModuleName.NRWAL,
                                        job_name=name, hardware='slurm',
                                        subprocess_manager=slurm_manager)

    msg = 'NRWAL CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'
               .format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running reV-NRWAL on SLURM with '
                    'node name "{}"'.format(name))
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path,
                                   conda_env=conda_env,
                                   module=module)[0]
        if out:
            msg = ('Kicked off reV-NRWAL job "{}" (SLURM jobid #{}).'
                   .format(name, out))

        Status.add_job(
            out_dir, module=ModuleName.NRWAL, job_name=name, replace=True,
            job_attrs={'job_id': out, 'hardware': 'slurm',
                       'fout': '{}.csv'.format(name), 'dirout': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV-NRWAL CLI.')
        raise
