# -*- coding: utf-8 -*-
"""
reV generation module.
"""
import copy
import logging
import numpy as np
import os
import pprint
import json

from reV.generation.base import BaseGen
from reV.utilities.exceptions import ProjectPointsValueError
from reV.SAM.generation import (Pvwattsv5, Pvwattsv7, TcsMoltenSalt, WindPower,
                                SolarWaterHeat, TroughPhysicalHeat,
                                LinearDirectSteam)

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)

ATTR_DIR = os.path.dirname(os.path.realpath(__file__))
ATTR_DIR = os.path.join(ATTR_DIR, 'output_attributes')
with open(os.path.join(ATTR_DIR, 'other.json'), 'r') as f:
    OTHER_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'generation.json'), 'r') as f:
    GEN_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'linear_fresnel.json'), 'r') as f:
    LIN_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'solar_water_heat.json'), 'r') as f:
    SWH_ATTRS = json.load(f)
with open(os.path.join(ATTR_DIR, 'trough_heat.json'), 'r') as f:
    TPPH_ATTRS = json.load(f)


class Gen(BaseGen):
    """reV generation analysis class to run SAM simulations"""

    # Mapping of reV technology strings to SAM generation objects
    OPTIONS = {'pvwattsv5': Pvwattsv5,
               'pvwattsv7': Pvwattsv7,
               'tcsmoltensalt': TcsMoltenSalt,
               'solarwaterheat': SolarWaterHeat,
               'troughphysicalheat': TroughPhysicalHeat,
               'lineardirectsteam': LinearDirectSteam,
               'windpower': WindPower,
               }

    # Mapping of reV generation outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = copy.deepcopy(OTHER_ATTRS)
    OUT_ATTRS.update(GEN_ATTRS)
    OUT_ATTRS.update(LIN_ATTRS)
    OUT_ATTRS.update(SWH_ATTRS)
    OUT_ATTRS.update(TPPH_ATTRS)
    OUT_ATTRS.update(BaseGen.ECON_ATTRS)

    def __init__(self, points_control, res_file, output_request=('cf_mean',),
                 site_data=None, out_fpath=None, drop_leap=False,
                 mem_util_lim=0.4):
        """
        Parameters
        ----------
        points_control : reV.config.project_points.PointsControl
            Project points control instance for site and SAM config spec.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        out_fpath : str, optional
            Output .h5 file path, by default None
        drop_leap : bool
            Drop leap day instead of final day of year during leap years
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        """

        super().__init__(points_control, output_request, site_data=site_data,
                         out_fpath=out_fpath, drop_leap=drop_leap,
                         mem_util_lim=mem_util_lim)

        self._res_file = res_file
        self._sam_module = self.OPTIONS[self.tech]
        self._run_attrs['sam_module'] = self._sam_module.MODULE
        self._run_attrs['res_file'] = res_file

        self._multi_h5_res, self._hsds = check_res_file(res_file)

        if self.tech not in self.OPTIONS:
            msg = ('Requested technology "{}" is not available. '
                   'reV generation can analyze the following '
                   'SAM technologies: {}'
                   .format(self.tech, list(self.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        # initialize output file
        self._init_fpath()
        self._init_h5()
        self._init_out_arrays()

    @property
    def res_file(self):
        """Get the resource filename and path.

        Returns
        -------
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        """
        return self._res_file

    @property
    def meta(self):
        """Get resource meta for all sites in project points.

        Returns
        -------
        meta : pd.DataFrame
            Meta data df for sites in project points. Column names are meta
            data variables, rows are different sites. The row index
            does not indicate the site number if the project points are
            non-sequential or do not start from 0, so a 'gid' column is added.
        """
        if self._meta is None:
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {'hsds': self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                res_meta = res.meta

            if np.max(self.project_points.sites) > len(res_meta):
                msg = ('ProjectPoints has a max site gid of {} which is '
                       'out of bounds for the meta data of size {} from '
                       'resource file: {}'
                       .format(np.max(self.project_points.sites),
                               res_meta.shape, self.res_file))
                logger.error(msg)
                raise ProjectPointsValueError(msg)

            self._meta = res_meta.iloc[self.project_points.sites, :]
            self._meta.loc[:, 'gid'] = self.project_points.sites
            self._meta.loc[:, 'reV_tech'] = self.project_points.tech

        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data.

        Returns
        -------
        _time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        if self._time_index is None:
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {'hsds': self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                time_index = res.time_index

            downscale = self.project_points.sam_config_obj.downscale
            step = self.project_points.sam_config_obj.time_index_step
            if downscale is not None:
                from rex.utilities.downscale import make_time_index
                year = time_index.year[0]
                ds_freq = downscale['frequency']
                time_index = make_time_index(year, ds_freq)
                logger.info('reV solar generation running with temporal '
                            'downscaling frequency "{}" with final '
                            'time_index length {}'
                            .format(ds_freq, len(time_index)))
            elif step is not None:
                time_index = time_index[::step]

            self._time_index = self.handle_leap_ti(time_index,
                                                   drop_leap=self._drop_leap)

        return self._time_index

    @classmethod
    def run(cls, points_control, tech=None, res_file=None, output_request=None,
            scale_outputs=True):
        """Run a SAM generation analysis based on the points_control iterator.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            A PointsControl instance dictating what sites and configs are run.
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.

        Returns
        -------
        out : dict
            Output dictionary from the SAM reV_run function. Data is scaled
            within this function to the datatype specified in Gen.OUT_ATTRS.
        """

        # Extract the site df from the project points df.
        site_df = points_control.project_points.df
        site_df = site_df.set_index('gid', drop=True)

        # run generation method for specified technology
        try:
            out = cls.OPTIONS[tech].reV_run(points_control, res_file, site_df,
                                            output_request=output_request)
        except Exception as e:
            out = {}
            logger.exception('Worker failed for PC: {}'.format(points_control))
            raise e

        if scale_outputs:
            # dtype convert in-place so no float data is stored unnecessarily
            for site, site_output in out.items():
                for k in site_output.keys():
                    # iterate through variable names in each site's output dict
                    if k in cls.OUT_ATTRS:
                        # get dtype and scale for output variable name
                        dtype = cls.OUT_ATTRS[k].get('dtype', 'float32')
                        scale_factor = cls.OUT_ATTRS[k].get('scale_factor', 1)

                        # apply scale factor and dtype
                        out[site][k] *= scale_factor
                        if np.issubdtype(dtype, np.integer):
                            # round after scaling if integer dtype
                            out[site][k] = np.round(out[site][k])

                        if isinstance(out[site][k], np.ndarray):
                            # simple astype for arrays
                            out[site][k] = out[site][k].astype(dtype)
                        else:
                            # use numpy array conversion for scalar values
                            out[site][k] = np.array([out[site][k]],
                                                    dtype=dtype)[0]

        return out

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : list
            Output variables requested from SAM.
        """

        output_request = self._output_request_type_check(req)

        # ensure that cf_mean is requested from output
        if 'cf_mean' not in output_request:
            output_request.append('cf_mean')

        for request in output_request:
            if request not in self.OUT_ATTRS:
                msg = ('User output request "{}" not recognized. '
                       'Will attempt to extract from PySAM.'.format(request))
                logger.debug(msg)

        return list(set(output_request))

    @classmethod
    def reV_run(cls, tech, points, sam_configs, res_file,
                output_request=('cf_mean',), site_data=None, curtailment=None,
                max_workers=1, sites_per_worker=None,
                pool_size=(os.cpu_count() * 2), timeout=1800,
                points_range=None, out_fpath=None, mem_util_lim=0.4,
                scale_outputs=True):
        """Execute a parallel reV generation run with smart data flushing.

        Parameters
        ----------
        tech : str
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed.
        points : slice | list | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_configs : dict | str | SAMConfig
            SAM input configuration ID(s) and file path(s). Keys are the SAM
            config ID(s) which map to the config column in the project points
            CSV. Values are either a JSON SAM config file or dictionary of SAM
            config inputs. Can also be a single config file path or a
            pre loaded SAMConfig object.
        res_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        output_request : list | tuple
            Output variables requested from SAM.
        site_data : str | pd.DataFrame | None
            Site-specific input data for SAM calculation. String should be a
            filepath that points to a csv, DataFrame is pre-extracted data.
            Rows match sites, columns are input keys. Need a "gid" column.
            Input as None if no site-specific data.
        curtailment : NoneType | dict | str | config.curtailment.Curtailment
            Inputs for curtailment parameters. If not None, curtailment inputs
            are expected. Can be:
                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config json file with path (str)
                - Instance of curtailment config object
                  (config.curtailment.Curtailment)
        max_workers : int
            Number of local workers to run on.
        sites_per_worker : int | None
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
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
        out_fpath : str, optional
            Output .h5 file path, by default None
        mem_util_lim : float
            Memory utilization limit (fractional). This will determine how many
            site results are stored in memory at any given time.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.

        Returns
        -------
        gen : Gen
            Gen instance with outputs saved to gen.out dict
        """

        # get a points control instance
        pc = cls.get_pc(points, points_range, sam_configs, tech,
                        sites_per_worker=sites_per_worker, res_file=res_file,
                        curtailment=curtailment)

        # make a Gen class instance to operate with
        gen = cls(pc, res_file,
                  output_request=output_request,
                  site_data=site_data,
                  out_fpath=out_fpath,
                  mem_util_lim=mem_util_lim)

        kwargs = {'tech': gen.tech,
                  'res_file': gen.res_file,
                  'output_request': gen.output_request,
                  'scale_outputs': scale_outputs}

        logger.info('Running reV generation for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_configs, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))

        # use serial or parallel execution control based on max_workers
        try:
            if max_workers == 1:
                logger.debug('Running serial generation for: {}'.format(pc))
                for pc_sub in pc:
                    gen.out = gen.run(pc_sub, **kwargs)

                gen.flush()
            else:
                logger.debug('Running parallel generation for: {}'.format(pc))
                gen._parallel_run(max_workers=max_workers, pool_size=pool_size,
                                  timeout=timeout, **kwargs)

        except Exception as e:
            logger.exception('reV generation failed!')
            raise e

        return gen
