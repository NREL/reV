# -*- coding: utf-8 -*-
"""
reV generation module.
"""
import os
import copy
import json
import logging
from warnings import warn

import pprint
import numpy as np
import pandas as pd

from reV.generation.base import BaseGen
from reV.utilities.exceptions import (ProjectPointsValueError, InputError,
                                      ConfigError, ConfigWarning)
from reV.SAM.generation import (Geothermal,
                                PvWattsv5,
                                PvWattsv7,
                                PvWattsv8,
                                PvSamv1,
                                TcsMoltenSalt,
                                WindPower,
                                SolarWaterHeat,
                                TroughPhysicalHeat,
                                LinearDirectSteam,
                                MhkWave)
from reV.utilities import ModuleName

from rex.resource import Resource
from rex.multi_file_resource import MultiFileResource
from rex.multi_res_resource import MultiResolutionResource
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
    >>> gen = Gen(sam_tech, sites, fp_sam, fp_res)
    >>> gen.run()
    >>>
    >>> gen.out
    {'cf_mean': array([0.16966143], dtype=float32)}
    >>>
    >>> sites = [3, 4, 7, 9]
    >>> req = ('cf_mean', 'cf_profile', 'lcoe_fcr')
    >>> gen = Gen(sam_tech, sites, fp_sam, fp_res, output_request=req)
    >>> gen.run()
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
    OPTIONS = {'geothermal': Geothermal,
               'pvwattsv5': PvWattsv5,
               'pvwattsv7': PvWattsv7,
               'pvwattsv8': PvWattsv8,
               'pvsamv1': PvSamv1,
               'tcsmoltensalt': TcsMoltenSalt,
               'solarwaterheat': SolarWaterHeat,
               'troughphysicalheat': TroughPhysicalHeat,
               'lineardirectsteam': LinearDirectSteam,
               'windpower': WindPower,
               'mhkwave': MhkWave
               }
    """reV technology options."""

    # Mapping of reV generation outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = copy.deepcopy(OTHER_ATTRS)
    OUT_ATTRS.update(GEN_ATTRS)
    OUT_ATTRS.update(LIN_ATTRS)
    OUT_ATTRS.update(SWH_ATTRS)
    OUT_ATTRS.update(TPPH_ATTRS)
    OUT_ATTRS.update(BaseGen.ECON_ATTRS)

    def __init__(self, technology, project_points, sam_files, resource_file,
                 low_res_resource_file=None, output_request=('cf_mean',),
                 site_data=None, curtailment=None, gid_map=None,
                 drop_leap=False, sites_per_worker=None, mem_util_lim=0.4,
                 scale_outputs=True, write_mapped_gids=False,
                 bias_correct=None):
        """
        Parameters
        ----------
        technology : str, optional
            SAM technology to analyze (pvwattsv7, windpower, tcsmoltensalt,
            solarwaterheat, troughphysicalheat, lineardirectsteam)
            The string should be lower-cased with spaces and _ removed,
            by default None
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
        resource_file : str
            Filepath to single resource file, multi-h5 directory,
            or /h5_dir/prefix*suffix
        low_res_resource_file : str | None
            Optional low resolution resource file that will be dynamically
            mapped+interpolated to the nominal-resolution res_file. This
            needs to be of the same format as resource_file, e.g. they both
            need to be handled by the same rex Resource handler such as
            WindResource
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
        gid_map : None | str | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids.  This can be None, a pre-extracted dict, or a
            filepath to json or csv. If this is a csv, it must have the columns
            "gid" (which matches the project points) and "gid_map" (gids to
            extract from the resource input)
        drop_leap : bool
            Drop leap day instead of final day of year during leap years
        sites_per_worker : int | None
            Number of sites to run in series on a worker. None defaults to the
            resource file chunk size.
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        write_mapped_gids : bool
            Option to write mapped gids to output meta instead of resource
            gids.
        bias_correct : str | pd.DataFrame
            Optional DataFrame or csv filepath to a wind or solar resource bias
            correction table. This has columns: gid (can be index name), adder,
            scalar. If both adder and scalar are present, the wind or solar
            resource is corrected by (res*scalar)+adder. If either is not
            present, scalar defaults to 1 and adder to 0. Only windspeed or
            GHI+DNI are corrected depending on the technology. GHI and DNI are
            corrected with the same correction factors.
        """
        pc = self.get_pc(points=project_points, points_range=None,
                         sam_configs=sam_files, tech=technology,
                         sites_per_worker=sites_per_worker,
                         res_file=resource_file,
                         curtailment=curtailment)

        super().__init__(pc, output_request, site_data=site_data,
                         drop_leap=drop_leap, mem_util_lim=mem_util_lim,
                         scale_outputs=scale_outputs)

        if self.tech not in self.OPTIONS:
            msg = ('Requested technology "{}" is not available. '
                   'reV generation can analyze the following '
                   'SAM technologies: {}'
                   .format(self.tech, list(self.OPTIONS.keys())))
            logger.error(msg)
            raise KeyError(msg)

        self.write_mapped_gids = write_mapped_gids
        self._res_file = resource_file
        self._lr_res_file = low_res_resource_file
        self._sam_module = self.OPTIONS[self.tech]
        self._run_attrs['sam_module'] = self._sam_module.MODULE
        self._run_attrs['res_file'] = resource_file

        self._multi_h5_res, self._hsds = check_res_file(resource_file)
        self._gid_map = self._parse_gid_map(gid_map)
        self._nn_map = self._parse_nn_map()
        self._bc = self._parse_bc(bias_correct)

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
    def lr_res_file(self):
        """Get the (optional) low-resolution resource filename and path.

        Returns
        -------
        str | None
        """
        return self._lr_res_file

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
            res_cls = Resource
            kwargs = {'hsds': self._hsds}
            if self._multi_h5_res:
                res_cls = MultiFileResource
                kwargs = {}

            res_gids = self.project_points.sites
            if self._gid_map is not None:
                res_gids = [self._gid_map[i] for i in res_gids]

            with res_cls(self.res_file, **kwargs) as res:
                meta_len = res.shapes['meta'][0]

                if np.max(res_gids) > meta_len:
                    msg = ('ProjectPoints has a max site gid of {} which is '
                           'out of bounds for the meta data of len {} from '
                           'resource file: {}'
                           .format(np.max(res_gids),
                                   meta_len, self.res_file))
                    logger.error(msg)
                    raise ProjectPointsValueError(msg)

                self._meta = res['meta', res_gids]

            self._meta.loc[:, 'gid'] = res_gids
            if self.write_mapped_gids:
                self._meta.loc[:, 'gid'] = self.project_points.sites
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
    def _run_single_worker(cls, points_control, tech=None, res_file=None,
                           lr_res_file=None, output_request=None,
                           scale_outputs=True, gid_map=None, nn_map=None,
                           bias_correct=None):
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
        lr_res_file : str | None
            Optional low resolution resource file that will be dynamically
            mapped+interpolated to the nominal-resolution res_file. This
            needs to be of the same format as resource_file, e.g. they both
            need to be handled by the same rex Resource handler such as
            WindResource
        output_request : list | tuple
            Output variables requested from SAM.
        scale_outputs : bool
            Flag to scale outputs in-place immediately upon Gen returning data.
        gid_map : None | dict
            Mapping of unique integer generation gids (keys) to single integer
            resource gids (values). This enables the user to input unique
            generation gids in the project points that map to non-unique
            resource gids. This can be None or a pre-extracted dict.
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings associated with the
            res_file to lr_res_file spatial mapping. For details on this
            argument, see the rex.MultiResolutionResource docstring.
        bias_correct : None | pd.DataFrame
            None if not provided or extracted DataFrame with wind or solar
            resource bias correction table. This has columns: gid (can be index
            name), adder, scalar. If both adder and scalar are present, the
            wind or solar resource is corrected by (res*scalar)+adder. If
            either adder or scalar is not present, scalar defaults to 1 and
            adder to 0. Only windspeed or GHI+DNI are corrected depending on
            the technology. GHI and DNI are corrected with the same correction
            factors.

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
            out = cls.OPTIONS[tech].reV_run(
                points_control, res_file, site_df,
                lr_res_file=lr_res_file,
                output_request=output_request,
                gid_map=gid_map, nn_map=nn_map,
                bias_correct=bias_correct)

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

    def _parse_nn_map(self):
        """Parse a nearest-neighbor spatial mapping array if lr_res_file is
        provided (resource data is at two resolutions and the low-resolution
        data must be mapped to the nominal-resolution data)

        Returns
        -------
        nn_map : np.ndarray
            Optional 1D array of nearest neighbor mappings associated with the
            res_file to lr_res_file spatial mapping. For details on this
            argument, see the rex.MultiResolutionResource docstring.
        """
        nn_map = None
        if self.lr_res_file is not None:

            handler_class = Resource
            if '*' in self.res_file or '*' in self.lr_res_file:
                handler_class = MultiFileResource

            with handler_class(self.res_file) as hr_res:
                with handler_class(self.lr_res_file) as lr_res:
                    logger.info('Making nearest neighbor map for multi '
                                'resolution resource data...')
                    nn_d, nn_map = MultiResolutionResource.make_nn_map(hr_res,
                                                                       lr_res)
                    logger.info('Done making nearest neighbor map for multi '
                                'resolution resource data!')

            logger.info('Made nearest neighbor mapping between nominal-'
                        'resolution and low-resolution resource files. '
                        'Min / mean / max dist: {:.3f} / {:.3f} / {:.3f}'
                        .format(nn_d.min(), nn_d.mean(), nn_d.max()))

        return nn_map

    @staticmethod
    def _parse_bc(bias_correct):
        """Parse the bias correction data.

        Parameters
        ----------
        bias_correct : str | pd.DataFrame | None
            Optional DataFrame or csv filepath to a wind or solar resource bias
            correction table. This has columns: gid (can be index name), adder,
            scalar. If both adder and scalar are present, the wind or solar
            resource is corrected by (res*scalar)+adder. If either is not
            present, scalar defaults to 1 and adder to 0. Only windspeed or
            GHI+DNI are corrected depending on the technology. GHI and DNI are
            corrected with the same correction factors.

        Returns
        -------
        bias_correct : None | pd.DataFrame
            None if not provided or extracted DataFrame with wind or solar
            resource bias correction table. This has columns: gid (can be index
            name), adder, scalar. If both adder and scalar are present, the
            wind or solar resource is corrected by (res*scalar)+adder. If
            either adder or scalar is not present, scalar defaults to 1 and
            adder to 0. Only windspeed or GHI+DNI are corrected depending on
            the technology. GHI and DNI are corrected with the same correction
            factors.
        """

        if isinstance(bias_correct, type(None)):
            return bias_correct

        elif isinstance(bias_correct, str):
            bias_correct = pd.read_csv(bias_correct)

        msg = ('Bias correction data must be a filepath to csv or a dataframe '
               'but received: {}'.format(type(bias_correct)))
        assert isinstance(bias_correct, pd.DataFrame), msg

        if 'adder' not in bias_correct:
            logger.info('Bias correction table provided, but "adder" not '
                        'found, defaulting to 0.')
            bias_correct['adder'] = 0

        if 'scalar' not in bias_correct:
            logger.info('Bias correction table provided, but "scalar" not '
                        'found, defaulting to 1.')
            bias_correct['scalar'] = 1

        msg = ('Bias correction table must have "gid" column but only found: '
               '{}'.format(list(bias_correct.columns)))
        assert 'gid' in bias_correct or bias_correct.index.name == 'gid', msg

        if bias_correct.index.name != 'gid':
            bias_correct = bias_correct.set_index('gid')

        return bias_correct

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

    def run(self, out_fpath=None, max_workers=1, timeout=1800,
            pool_size=(os.cpu_count() * 2)):
        """Execute a parallel reV generation run with smart data flushing.

        Parameters
        ----------
        out_fpath : str, optional
            Path to output file. If ``None``, no output file will
            be written. If the filepath is specified but the module name
            (generation) and/or resource data year is not included, the
            module name and/or resource data year will get added to the
            output file name. By default, ``None``.
        max_workers : int, optional
            Number of local workers to run on. By default, ``1``.
        timeout : int, optional
            Number of seconds to wait for parallel run iteration to
            complete before returning zeros. By default, ``1800``
            seconds.
        pool_size : tuple, optional
            Number of futures to submit to a single process pool for
            parallel futures. By default, ``(os.cpu_count() * 2)``.

        Returns
        -------
        str | None
            Path to output HDF5 file, or ``None`` if results were not
            written to disk.
        """
        # initialize output file
        self._init_fpath(out_fpath, module=ModuleName.GENERATION)
        self._init_h5()
        self._init_out_arrays()

        kwargs = {'tech': self.tech,
                  'res_file': self.res_file,
                  'lr_res_file': self.lr_res_file,
                  'output_request': self.output_request,
                  'scale_outputs': self.scale_outputs,
                  'gid_map': self._gid_map,
                  'nn_map': self._nn_map,
                  'bias_correct': self._bc}

        logger.info('Running reV generation for: {}'
                    .format(self.points_control))
        logger.debug('The following project points were specified: "{}"'
                     .format(self.project_points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(self.sam_configs, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(self.output_request))

        # use serial or parallel execution control based on max_workers
        try:
            if max_workers == 1:
                logger.debug('Running serial generation for: {}'
                             .format(self.points_control))
                for i, pc_sub in enumerate(self.points_control):
                    self.out = self._run_single_worker(pc_sub, **kwargs)
                    logger.info('Finished reV gen serial compute for: {} '
                                '(iteration {} out of {})'
                                .format(pc_sub, i + 1,
                                        len(self.points_control)))
                self.flush()
            else:
                logger.debug('Running parallel generation for: {}'
                             .format(self.points_control))
                self._parallel_run(max_workers=max_workers,
                                   pool_size=pool_size, timeout=timeout,
                                   **kwargs)

        except Exception as e:
            logger.exception('reV generation failed!')
            raise e

        return self._out_fpath


# TODO: Move this into gen CLI file
# TODO: Add logging
def gen_preprocessor(config, out_dir, job_name, analysis_years=None):
    """Preprocess generation config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.
    job_name : str
        Name of bespoke job. This will be included in the output file
        name.
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
    if not isinstance(analysis_years, list):
        analysis_years = [analysis_years]

    if analysis_years[0] is None:
        warn('Years may not have been specified, may default '
             'to available years in inputs files.', ConfigWarning)

    config["resource_file"] = _parse_res_files(config["resource_file"],
                                               analysis_years)
    lr_res_file = config.get("low_res_resource_file")
    if lr_res_file is None:
        config["low_res_resource_file"] = [None] * len(analysis_years)
    else:
        config["low_res_resource_file"] = _parse_res_files(lr_res_file,
                                                           analysis_years)

    config['technology'] = (config['technology'].lower()
                            .replace(' ', '').replace('_', ''))
    config["out_fpath"] = os.path.join(out_dir, job_name)
    return config


def _parse_res_files(res_fps, analysis_years):
    """Parse the base resource file input into correct ordered list format
    with year imputed in the {} format string"""

    # get base filename, may have {} for year format
    if isinstance(res_fps, str) and '{}' in res_fps:
        # need to make list of res files for each year
        res_fps = [res_fps.format(year) for year in analysis_years]
    elif isinstance(res_fps, str):
        # only one resource file request, still put in list
        res_fps = [res_fps]
    elif not isinstance(res_fps, (list, tuple)):
        msg = ('Bad "resource_file" type, needed str, list, or tuple '
               'but received: {}, {}'
               .format(res_fps, type(res_fps)))
        logger.error(msg)
        raise ConfigError(msg)

    if len(res_fps) != len(analysis_years):
        msg = ('The number of resource files does not match '
               'the number of analysis years!'
               '\n\tResource files: \n\t\t{}'
               '\n\tYears: \n\t\t{}'
               .format(res_fps, analysis_years))
        logger.error(msg)
        raise ConfigError(msg)

    return res_fps
