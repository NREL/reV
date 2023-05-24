# -*- coding: utf-8 -*-
"""
reV econ module (lcoe-fcr, single owner, etc...)
"""
import logging
import numpy as np
import os
import pandas as pd
import pprint
from warnings import warn

from reV.config.project_points import PointsControl
from reV.generation.base import BaseGen
from reV.handlers.outputs import Outputs
from reV.SAM.econ import LCOE as SAM_LCOE
from reV.SAM.econ import SingleOwner
from reV.SAM.windbos import WindBos
from reV.utilities.exceptions import (ExecutionError, OffshoreWindInputWarning,
                                      ConfigError, ConfigWarning)
from reV.utilities import ModuleName

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.utilities import check_res_file, parse_year

from gaps.pipeline import parse_previous_status

logger = logging.getLogger(__name__)


class Econ(BaseGen):
    """reV econ analysis class to run SAM simulations"""

    # Mapping of reV econ output strings to SAM econ modules
    OPTIONS = {'lcoe_fcr': SAM_LCOE,
               'ppa_price': SingleOwner,
               'project_return_aftertax_npv': SingleOwner,
               'lcoe_real': SingleOwner,
               'lcoe_nom': SingleOwner,
               'flip_actual_irr': SingleOwner,
               'gross_revenue': SingleOwner,
               'total_installed_cost': WindBos,
               'turbine_cost': WindBos,
               'sales_tax_cost': WindBos,
               'bos_cost': WindBos,
               'fixed_charge_rate': SAM_LCOE,
               'capital_cost': SAM_LCOE,
               'fixed_operating_cost': SAM_LCOE,
               'variable_operating_cost': SAM_LCOE,
               }

    # Mapping of reV econ outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = BaseGen.ECON_ATTRS

    def __init__(self, project_points, sam_files, cf_file, site_data=None,
                 output_request=('lcoe_fcr',), sites_per_worker=100,
                 mem_util_lim=0.4, append=False):
        """Initialize an econ instance.

        Parameters
        ----------
        project_points : int | slice | list | tuple | str | pd.DataFrame | dict
            Slice specifying project points, string pointing to a project
            points csv, or a dataframe containing the effective csv contents.
            Can also be a single integer site value.
        sam_files : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        cf_file : str
            reV generation capacity factor output file with path.
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        sites_per_worker : int
            Number of sites to run in series on a worker.
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the out_fpath and dirout inputs.
        """

        # get a points control instance
        pc = self.get_pc(points=project_points, points_range=None,
                         sam_configs=sam_files, cf_file=cf_file,
                         sites_per_worker=sites_per_worker, append=append)

        super().__init__(pc, output_request, site_data=site_data,
                         mem_util_lim=mem_util_lim)

        self._cf_file = cf_file
        self._append = append
        self._run_attrs['cf_file'] = cf_file
        self._run_attrs['sam_module'] = self._sam_module.MODULE

    @property
    def cf_file(self):
        """Get the capacity factor output filename and path.

        Returns
        -------
        cf_file : str
            reV generation capacity factor output file with path.
        """
        return self._cf_file

    @property
    def meta(self):
        """Get meta data from the source capacity factors file.

        Returns
        -------
        _meta : pd.DataFrame
            Meta data from capacity factor outputs file.
        """
        if self._meta is None and self.cf_file is not None:
            with Outputs(self.cf_file) as cfh:
                # only take meta that belongs to this project's site list
                self._meta = cfh.meta[
                    cfh.meta['gid'].isin(self.points_control.sites)]

            if 'offshore' in self._meta:
                if self._meta['offshore'].sum() > 1:
                    w = ('Found offshore sites in econ meta data. '
                         'This functionality has been deprecated. '
                         'Please run the reV offshore module to '
                         'calculate offshore wind lcoe.')
                    warn(w, OffshoreWindInputWarning)
                    logger.warning(w)

        elif self._meta is None and self.cf_file is None:
            self._meta = pd.DataFrame({'gid': self.points_control.sites})

        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data."""
        if self._time_index is None and self.cf_file is not None:
            with Outputs(self.cf_file) as cfh:
                if 'time_index' in cfh.datasets:
                    self._time_index = cfh.time_index

        return self._time_index

    @staticmethod
    def _econ_append_pc(pp, cf_file, sites_per_worker=None):
        """
        Generate ProjectControls for econ append

        Parameters
        ----------
        pp : reV.config.project_points.ProjectPoints
            ProjectPoints to adjust gids for
        cf_file : str
            reV generation capacity factor output file with path.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        multi_h5_res, hsds = check_res_file(cf_file)
        if multi_h5_res:
            res_cls = MultiFileResource
            res_kwargs = {}
        else:
            res_cls = Resource
            res_kwargs = {'hsds': hsds}

        with res_cls(cf_file, **res_kwargs) as f:
            gid0 = f.meta['gid'].values[0]
            gid1 = f.meta['gid'].values[-1]

        i0 = pp.index(gid0)
        i1 = pp.index(gid1) + 1
        pc = PointsControl.split(i0, i1, pp, sites_per_split=sites_per_worker)

        return pc

    @classmethod
    def get_pc(cls, points, points_range, sam_configs, cf_file,
               sites_per_worker=None, append=False):
        """
        Get a PointsControl instance.

        Parameters
        ----------
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        cf_file : str
            reV generation capacity factor output file with path.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the out_fpath and dirout inputs.

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        pc = super().get_pc(points, points_range, sam_configs, ModuleName.ECON,
                            sites_per_worker=sites_per_worker,
                            res_file=cf_file)

        if append:
            pc = cls._econ_append_pc(pc.project_points, cf_file,
                                     sites_per_worker=sites_per_worker)

        return pc

    @staticmethod
    def _run_single_worker(pc, econ_fun, output_request, **kwargs):
        """Run the SAM econ calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
            Must have project_points with df property with all relevant
            site-specific inputs and a 'gid' column. By passing site-specific
            inputs in this dataframe, which was split using points_control,
            only the data relevant to the current sites is passed.
        econ_fun : method
            reV_run() method from one of the econ modules (SingleOwner,
            SAM_LCOE, WindBos).
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        kwargs : dict
            Additional input parameters for the SAM run module.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in Econ.OUT_ATTRS.
        """

        # make sure output request is a list
        if isinstance(output_request, str):
            output_request = [output_request]

        # Extract the site df from the project points df.
        site_df = pc.project_points.df
        site_df = site_df.set_index('gid', drop=True)

        # SAM execute econ analysis based on output request
        try:
            out = econ_fun(pc, site_df, output_request=output_request,
                           **kwargs)
        except Exception as e:
            out = {}
            logger.exception('Worker failed for PC: {}'.format(pc))
            raise e

        return out

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : str| list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

        output_request = self._output_request_type_check(req)

        for request in output_request:
            if request not in self.OUT_ATTRS:
                msg = ('User output request "{}" not recognized. '
                       'Will attempt to extract from PySAM.'.format(request))
                logger.debug(msg)

        modules = []
        for request in output_request:
            if request in self.OPTIONS:
                modules.append(self.OPTIONS[request])

        if not any(modules):
            msg = ('None of the user output requests were recognized. '
                   'Cannot run reV econ. '
                   'At least one of the following must be requested: {}'
                   .format(list(self.OPTIONS.keys())))
            logger.exception(msg)
            raise ExecutionError(msg)

        b1 = [m == modules[0] for m in modules]
        b2 = np.array([m == WindBos for m in modules])
        b3 = np.array([m == SingleOwner for m in modules])

        if all(b1):
            self._sam_module = modules[0]
            self._fun = modules[0].reV_run
        elif all(b2 | b3):
            self._sam_module = SingleOwner
            self._fun = SingleOwner.reV_run
        else:
            msg = ('Econ outputs requested from different SAM modules not '
                   'currently supported. Output request variables require '
                   'SAM methods: {}'.format(modules))
            raise ValueError(msg)

        return list(set(output_request))

    def _get_data_shape(self, dset, n_sites):
        """Get the output array shape based on OUT_ATTRS or PySAM.Outputs.

        This Econ get data shape method will also first check for the dset in
        the site_data table. If not found in site_data, the dataset will be
        looked for in OUT_ATTRS and PySAM.Outputs as it would for Generation.

        Parameters
        ----------
        dset : str
            Variable name to get shape for.
        n_sites : int
            Number of sites for this data shape.

        Returns
        -------
        shape : tuple
            1D or 2D shape tuple for dset.
        """

        if dset in self.site_data:
            data_shape = (n_sites, )
            data = self.site_data[dset].values[0]

            if isinstance(data, (list, tuple, np.ndarray, str)):
                msg = ('Cannot pass through non-scalar site_data '
                       'input key "{}" as an output_request!'.format(dset))
                logger.error(msg)
                raise ExecutionError(msg)

        else:
            data_shape = super()._get_data_shape(dset, n_sites)

        return data_shape

    def reV_run(self, out_dir=None, max_workers=1, timeout=1800,
                pool_size=(os.cpu_count() * 2), job_name=None):
        """Execute a parallel reV econ run with smart data flushing.

        Parameters
        ----------
        out_dir : str, optional
            Path to output directory. If ``None``, no output file will
            be written. If this class was initialized with
            ``append=True``, this option has no effect.
            By default, ``None``.
        max_workers : int, optional
            Number of local workers to run on. By default, ``1``.
        timeout : int, optional
            Number of seconds to wait for parallel run iteration to
            complete before returning zeros. By default, ``1800``
            seconds.
        pool_size : tuple, optional
            Number of futures to submit to a single process pool for
            parallel futures. By default, ``(os.cpu_count() * 2)``.
        job_name : str, optional
            Name for job. This string will be incorporated into the reV
            generation output file name. If ``None``, the module name
            (econ) will be used. If this class was initialized with
            ``append=True``, this option has no effect.
            By default, ``None``.

        Returns
        -------
        str | None
            Path to output HDF5 file, or ``None`` if results were not
            written to disk.
        """

        # initialize output file or append econ data to gen file
        if self._append:
            self._out_fpath = self._cf_file
        else:
            if out_dir is not None:
                out_dir = os.path.join(out_dir, job_name or ModuleName.ECON)
            self._init_fpath(out_dir, ModuleName.ECON)

        self._init_h5(mode='a' if self._append else 'w')
        self._init_out_arrays()

        diff = list(set(self.points_control.sites)
                    - set(self.meta['gid'].values))
        if diff:
            raise Exception('The following analysis sites were requested '
                            'through project points for econ but are not '
                            'found in the CF file ("{}"): {}'
                            .format(self.cf_file, diff))

        # make a kwarg dict
        kwargs = {'output_request': self.output_request,
                  'cf_file': self.cf_file,
                  'year': self.year}

        logger.info('Running econ with smart data flushing '
                    'for: {}'.format(self.points_control))
        logger.debug('The following project points were specified: "{}"'
                     .format(self.project_points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(self.sam_configs, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(self.output_request))

        try:
            kwargs['econ_fun'] = self._fun
            if max_workers == 1:
                logger.debug('Running serial econ for: {}'
                             .format(self.points_control))
                for i, pc_sub in enumerate(self.points_control):
                    self.out = self._run_single_worker(pc_sub, **kwargs)
                    logger.info('Finished reV econ serial compute for: {} '
                                '(iteration {} out of {})'
                                .format(pc_sub, i + 1,
                                        len(self.points_control)))
                self.flush()
            else:
                logger.debug('Running parallel econ for: {}'
                             .format(self.points_control))
                self._parallel_run(max_workers=max_workers,
                                   pool_size=pool_size, timeout=timeout,
                                   **kwargs)

        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed for econ.')
            raise e

        return self._out_fpath


def econ_preprocessor(config, project_dir, analysis_years=None):
    """Preprocess econ config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    analysis_years : int | list, optional
        A single year or list of years to perform analysis for. These
        years will be used to fill in any brackets ``{}`` in the
        ``resource_file`` input. If ``None``, the ``resource_file``
        input is assumed to be the full path to the single resource
        file to be processed.  By default, ``None``.

    Returns
    -------
    dict
        Updated config file.
    """
    # TODO: Keep it DRY with gen preprocessor
    if not isinstance(analysis_years, list):
        analysis_years = [analysis_years]

    if analysis_years[0] is None:
        warn('Years may not have been specified, may default '
             'to available years in inputs files.', ConfigWarning)

    config["cf_file"] = parse_cf_files(config["cf_file"],
                                       analysis_years, project_dir)

    return config


def parse_cf_files(cf_file, analysis_years, project_dir):
    """Get the capacity factor files (reV generation output data).

    Returns
    -------
    cf_files : list
        Target paths for capacity factor files (reV generation output
        data) for input to reV LCOE calculation.
    """
    # get base filename, may have {} for year format
    if '{}' in cf_file:
        # need to make list of res files for each year
        cf_files = [cf_file.format(year) for year in analysis_years]
    elif 'PIPELINE' in cf_file:
        cf_files = parse_previous_status(project_dir,
                                         command=str(ModuleName.ECON))
    else:
        # only one resource file request, still put in list
        cf_files = [cf_file]

    for f in cf_files:
        # ignore files that are to be specified using pipeline utils
        if 'PIPELINE' not in os.path.basename(f):
            if not os.path.exists(f):
                raise IOError('File does not exist: {}'.format(f))

    # check year/cf_file matching if not a pipeline input
    if 'PIPELINE' not in cf_file:
        if len(cf_files) != len(analysis_years):
            raise ConfigError('The number of cf files does not match '
                              'the number of analysis years!'
                              '\n\tCF files: \n\t\t{}'
                              '\n\tYears: \n\t\t{}'
                              .format(cf_files, analysis_years))
        for year in analysis_years:
            if str(year) not in str(cf_files):
                raise ConfigError('Could not find year {} in cf '
                                  'files: {}'.format(year, cf_files))

    return [fn for fn in cf_files if parse_year(fn) in analysis_years]
