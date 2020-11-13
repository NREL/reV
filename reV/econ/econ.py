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
from reV.utilities.exceptions import ExecutionError, OffshoreWindInputWarning

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.utilities import check_res_file

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
               }

    # Mapping of reV econ outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = {'other': {'scale_factor': 1, 'units': 'unknown',
                           'dtype': 'float32', 'chunks': None},
                 'lcoe_fcr': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 'ppa_price': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'project_return_aftertax_npv': {'scale_factor': 1,
                                                 'units': 'dol',
                                                 'dtype': 'float32',
                                                 'chunks': None,
                                                 'type': 'scalar'},
                 'lcoe_real': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'lcoe_nom': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 'flip_actual_irr': {'scale_factor': 1, 'units': 'perc',
                                     'dtype': 'float32', 'chunks': None,
                                     'type': 'scalar'},
                 'gross_revenue': {'scale_factor': 1, 'units': 'dollars',
                                   'dtype': 'float32', 'chunks': None,
                                   'type': 'scalar'},
                 'total_installed_cost': {'scale_factor': 1,
                                          'units': 'dollars',
                                          'dtype': 'float32', 'chunks': None,
                                          'type': 'scalar'},
                 'turbine_cost': {'scale_factor': 1, 'units': 'dollars',
                                  'dtype': 'float32', 'chunks': None,
                                  'type': 'scalar'},
                 'sales_tax_cost': {'scale_factor': 1, 'units': 'dollars',
                                    'dtype': 'float32', 'chunks': None,
                                    'type': 'scalar'},
                 'bos_cost': {'scale_factor': 1, 'units': 'dollars',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 }

    def __init__(self, points_control, cf_file, year, site_data=None,
                 output_request=('lcoe_fcr',), fout=None, dirout='./econ_out',
                 append=False, mem_util_lim=0.4):
        """Initialize an econ instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            Project points control instance for site and SAM config spec.
        cf_file : str
            reV generation capacity factor output file with path.
        year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the fout and dirout inputs.
        """

        super().__init__(points_control, output_request, site_data=site_data,
                         fout=fout, dirout=dirout, mem_util_lim=mem_util_lim)

        self._cf_file = cf_file
        self._year = year
        self._run_attrs['cf_file'] = cf_file
        self._run_attrs['sam_module'] = self._sam_module.MODULE

        # initialize output file or append econ data to gen file
        if append:
            self._fpath = self._cf_file
        else:
            self._init_fpath()

        mode = 'a' if append else 'w'
        self._init_h5(mode=mode)
        self._init_out_arrays()

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
    def get_pc(cls, points, points_range, sam_files, cf_file,
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
        sam_files : dict | str | list | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s), top level value is the SAM path. Can also be a single
            config file str. If it's a list, it is mapped to the sorted list
            of unique configs requested by points csv. Can also be a
            pre loaded SAMConfig object.
        cf_file : str
            reV generation capacity factor output file with path.
        sites_per_worker : int
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the fout and dirout inputs.

        Returns
        -------
        pc : reV.config.project_points.PointsControl
            PointsControl object instance.
        """
        pc = super().get_pc(points, points_range, sam_files, 'econ',
                            sites_per_worker=sites_per_worker,
                            res_file=cf_file)

        if append:
            pc = cls._econ_append_pc(pc.project_points, cf_file,
                                     sites_per_worker=sites_per_worker)

        return pc

    @staticmethod
    def run(pc, econ_fun, output_request, **kwargs):
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

    @classmethod
    def reV_run(cls, points, sam_files, cf_file,
                year=None, site_data=None, output_request=('lcoe_fcr',),
                max_workers=1, sites_per_worker=100,
                pool_size=(os.cpu_count() * 2),
                timeout=1800, points_range=None, fout=None,
                dirout='./econ_out', append=False):
        """Execute a parallel reV econ run with smart data flushing.

        Parameters
        ----------
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Site-agnostic input data.
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        cf_file : str
            reV generation capacity factor output file with path.
        year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        max_workers : int
            Number of local workers to run on.
        sites_per_worker : int
            Number of sites to run in series on a worker.
        pool_size : int
            Number of futures to submit to a single process pool for
            parallel futures.
        timeout : int | float
            Number of seconds to wait for parallel run iteration to complete
            before returning zeros. Default is 1800 seconds.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification. None will return object.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        append : bool
            Flag to append econ datasets to source cf_file. This has priority
            over the fout and dirout inputs.

        Returns
        -------
        econ : Econ
            Econ object instance with outputs stored in econ.out dict.
        """

        # get a points control instance
        pc = cls.get_pc(points, points_range, sam_files, cf_file,
                        sites_per_worker=sites_per_worker, append=append)

        # make a class instance to operate with
        econ = cls(pc, cf_file, year=year, site_data=site_data,
                   output_request=output_request, fout=fout, dirout=dirout,
                   append=append)

        diff = list(set(pc.sites) - set(econ.meta['gid'].values))
        if diff:
            raise Exception('The following analysis sites were requested '
                            'through project points for econ but are not '
                            'found in the CF file ("{}"): {}'
                            .format(econ.cf_file, diff))

        # make a kwarg dict
        kwargs = {'output_request': econ.output_request,
                  'cf_file': econ.cf_file,
                  'year': econ.year}

        logger.info('Running econ with smart data flushing '
                    'for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_files, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))

        try:
            kwargs['econ_fun'] = econ._fun
            if max_workers == 1:
                logger.debug('Running serial econ for: {}'.format(pc))
                for pc_sub in pc:
                    econ.out = econ.run(pc_sub, **kwargs)
                econ.flush()
            else:
                logger.debug('Running parallel econ for: {}'.format(pc))
                econ._parallel_run(max_workers=max_workers,
                                   pool_size=pool_size, timeout=timeout,
                                   **kwargs)

        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed for econ.')
            raise e

        return econ
