# -*- coding: utf-8 -*-
"""
reV generation module.
"""
import copy
import logging
import numpy as np
import pandas as pd
import os
import pprint
import json

from reV.generation.base import BaseGen
from reV.utilities.exceptions import ProjectPointsValueError, InputError
from reV.SAM.generation import (PvWattsv5,
                                PvWattsv7,
                                PvSamv1,
                                TcsMoltenSalt,
                                WindPower,
                                SolarWaterHeat,
                                TroughPhysicalHeat,
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
    """reV generation analysis class to run SAM simulations


    Examples
    --------
    The following is an example of the most simple way to run reV generation.
    The reV code pipes in renewable energy resource data (usually from the
    NSRDB or WTK), loads the SAM config, and then executes the PySAM compute
    module for a given technology. If economic parameters are supplied, you can
    bundle a "follow-on" econ calculation by just adding the desired econ
    output keys to the output_request kwarg. You can request reV to run the
    analysis for one or more "sites", which correspond to the meta indices in
    the resource data (also commonly called the gid's). Note that the
    TESTDATADIR refers to the local cloned repository and will need to be
    replaced with a valid path if you installed reV via a simple pip install.

    >>> import os
    >>> from reV import Gen, TESTDATADIR
    >>>
    >>> sam_tech = 'pvwattsv7'
    >>> sites = 0
    >>> fp_sam = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')
    >>> fp_res = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2013.h5')
    >>>
    >>> gen = Gen.reV_run(sam_tech, sites, fp_sam, fp_res)
    >>>
    >>> gen.out
    {'cf_mean': array([0.16966143], dtype=float32)}
    >>>
    >>> sites = [3, 4, 7, 9]
    >>> req = ('cf_mean', 'cf_profile', 'lcoe_fcr')
    >>> gen = Gen.reV_run(sam_tech, sites, fp_sam, fp_res, output_request=req)
    >>>
    >>> gen.out
    {'lcoe_fcr': array([131.39166, 131.31221, 127.54539, 125.49656]),
     'cf_mean': array([0.17713654, 0.17724372, 0.1824783 , 0.1854574 ]),
     'cf_profile': array([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            ...,
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
    """

    # Mapping of reV technology strings to SAM generation objects
    OPTIONS = {'pvwattsv5': PvWattsv5,
               'pvwattsv7': PvWattsv7,
               'pvsamv1': PvSamv1,
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
                 site_data=None, gid_map=None, out_fpath=None, drop_leap=False,
                 mem_util_lim=0.4, scale_outputs=True):
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
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids.  This can be None, a pre-extracted dict, or a
            filepath to json or csv. If this is a csv, it must have the columns
            "gid" (which matches the project points) and "gid_map" (gids to
            extract from the resource input)
        out_fpath : str, optional
            Output .h5 file path, by default None
        drop_leap : bool
            Drop leap day instead of final day of year during leap years
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        """

        super().__init__(points_control, output_request, site_data=site_data,
                         out_fpath=out_fpath, drop_leap=drop_leap,
                         mem_util_lim=mem_util_lim,
                         scale_outputs=scale_outputs)

        if self.tech not in self.OPTIONS:
            msg = ('Requested technology "{}" is not available. '
                   'reV generation can analyze the following '
                   'SAM technologies: {}'
                   .format(self.tech, list(self.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        self._res_file = res_file
        self._sam_module = self.OPTIONS[self.tech]
        self._run_attrs['sam_module'] = self._sam_module.MODULE
        self._run_attrs['res_file'] = res_file

        self._multi_h5_res, self._hsds = check_res_file(res_file)
        self._gid_map = self._parse_gid_map(gid_map)

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

            res_gids = self.project_points.sites
            if self._gid_map is not None:
                res_gids = [self._gid_map[i] for i in res_gids]

            if np.max(res_gids) > len(res_meta):
                msg = ('ProjectPoints has a max site gid of {} which is '
                       'out of bounds for the meta data of size {} from '
                       'resource file: {}'
                       .format(np.max(res_gids),
                               res_meta.shape, self.res_file))
                logger.error(msg)
                raise ProjectPointsValueError(msg)

            self._meta = res_meta.iloc[res_gids, :]
            self._meta.loc[:, 'gid'] = res_gids
            self._meta.index = self.project_points.sites
            self._meta.index.name = 'gid'
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
            scale_outputs=True, gid_map=None):
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
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.

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
                                            output_request=output_request,
                                            gid_map=gid_map)
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

    def _parse_gid_map(self, gid_map):
        """
        Parameters
        ----------
        gid_map : None | dict | str
            This can be None, a pre-extracted dict, or a filepath to json or
            csv. If this is a csv, it must have the columns "gid" (which
            matches the project points) and "gid_map" (gids to extract from the
            resource input)

        Returns
        -------
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids.
        """

        if isinstance(gid_map, str):
            if gid_map.endswith('.csv'):
                gid_map = pd.read_csv(gid_map).to_dict()
                assert 'gid' in gid_map, 'Need "gid" in gid_map column'
                assert 'gid_map' in gid_map, 'Need "gid_map" in gid_map column'
                gid_map = {gid_map['gid'][i]: gid_map['gid_map'][i]
                           for i in gid_map['gid'].keys()}

            elif gid_map.endswith('.json'):
                with open(gid_map, 'r') as f:
                    gid_map = json.load(f)

        if isinstance(gid_map, dict):
            if not self._multi_h5_res:
                res_cls = Resource
                kwargs = {'hsds': self._hsds}
            else:
                res_cls = MultiFileResource
                kwargs = {}

            with res_cls(self.res_file, **kwargs) as res:
                for gen_gid, res_gid in gid_map.items():
                    msg1 = ('gid_map values must all be int but received '
                            '{}: {}'.format(gen_gid, res_gid))
                    msg2 = ('Could not find the gen_gid to res_gid mapping '
                            '{}: {} in the resource meta data.'
                            .format(gen_gid, res_gid))
                    assert isinstance(gen_gid, int), msg1
                    assert isinstance(res_gid, int), msg1
                    assert res_gid in res.meta.index.values, msg2

                for gen_gid in self.project_points.sites:
                    msg3 = ('Could not find the project points gid {} in the '
                            'gen_gid input of the gid_map.'.format(gen_gid))
                    assert gen_gid in gid_map, msg3

        elif gid_map is not None:
            msg = ('Could not parse gid_map, must be None, dict, or path to '
                   'csv or json, but received: {}'.format(gid_map))
            logger.error(msg)
            raise InputError(msg)

        return gid_map

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
                gid_map=None, max_workers=1, sites_per_worker=None,
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
        points : int | slice | list | str | PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object. Can
            also be a single site integer values.
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
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids.  This can be None, a pre-extracted dict, or a
            filepath to json or csv. If this is a csv, it must have the columns
            "gid" (which matches the project points) and "gid_map" (gids to
            extract from the resource input)
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
                  gid_map=gid_map,
                  out_fpath=out_fpath,
                  mem_util_lim=mem_util_lim,
                  scale_outputs=scale_outputs)

        kwargs = {'tech': gen.tech,
                  'res_file': gen.res_file,
                  'output_request': gen.output_request,
                  'scale_outputs': scale_outputs,
                  'gid_map': gen._gid_map,
                  }

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
                    logger.info('Finished reV gen serial compute for: {}'
                                .format(pc_sub))
                gen.flush()
            else:
                logger.debug('Running parallel generation for: {}'.format(pc))
                gen._parallel_run(max_workers=max_workers, pool_size=pool_size,
                                  timeout=timeout, **kwargs)

        except Exception as e:
            logger.exception('reV generation failed!')
            raise e

        return gen
