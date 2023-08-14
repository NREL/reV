# -*- coding: utf-8 -*-
"""
reV generation module.
"""
import os
import copy
import json
import logging

import pprint
import numpy as np
import pandas as pd

from reV.generation.base import BaseGen
from reV.utilities.exceptions import (ProjectPointsValueError, InputError)
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
    """Gen"""

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
                 drop_leap=False, sites_per_worker=None,
                 memory_utilization_limit=0.4, scale_outputs=True,
                 write_mapped_gids=False, bias_correct=None):
        """reV generation analysis class.

        ``reV`` generation analysis runs SAM simulations by piping in
        renewable energy resource data (usually from the NSRDB or WTK),
        loading the SAM config, and then executing the PySAM compute
        module for a given technology. If economic parameters are
        supplied, you can bundle a "follow-on" econ calculation by
        just adding the desired econ output keys to the `output_request`
        input. You can request ``reV`` to run the analysis for one or
        more "sites", which correspond to the meta indices in the
        resource data (also commonly called the ``gid's``).

        Examples
        --------
        The following is an example of the most simple way to run reV
        generation. Note that the ``TESTDATADIR`` refers to the local cloned
        repository and will need to be replaced with a valid path if you
        installed ``reV`` via a simple pip install.

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

        Parameters
        ----------
        technology : str
            String indicating which SAM technology to analyze. Must be
            one of the keys of
            :attr:`~reV.generation.generation.Gen.OPTIONS`. The string
            should be lower-cased with spaces and underscores removed.
        project_points : int | list | tuple | str | dict | pd.DataFrame | slice
            Input specifying which sites to process. A single integer
            representing the GID of a site may be specified to evaluate
            reV at a single location. A list or tuple of integers
            (or slice) representing the GIDs of multiple sites can be
            specified to evaluate reV at multiple specific locations.
            A string pointing to a project points CSV file may also be
            specified. Typically, the CSV contains two columns:

                - ``gid``: Integer specifying the GID of each site.
                - ``config``: Key in the `sam_files` input dictionary
                  (see below) corresponding to the SAM configuration to
                  use for each particular site. This value can also be
                  ``None`` (or left out completely) if you specify only
                  a single SAM configuration file as the `sam_files`
                  input.

            The CSV file may also contain site-specific inputs by
            including a column named after a config keyword (e.g. a
            column called ``capital_cost`` may be included to specify a
            site-specific capital cost value for each location). Columns
            that do not correspond to a config key may also be included,
            but they will be ignored. A DataFrame following the same
            guidelines as the CSV input (or a dictionary that can be
            used to initialize such a DataFrame) may be used for this
            input as well.
        sam_files : dict | str
            A dictionary mapping SAM input configuration ID(s) to SAM
            configuration(s). Keys are the SAM config ID(s) which
            correspond to the ``config`` column in the project points
            CSV. Values for each key are either a path to a
            corresponding SAM config file or a full dictionary
            of SAM config inputs. For example::

                sam_files = {
                    "default": "/path/to/default/sam.json",
                    "onshore": "/path/to/onshore/sam_config.yaml",
                    "offshore": {
                        "sam_key_1": "sam_value_1",
                        "sam_key_2": "sam_value_2",
                        ...
                    },
                    ...
                }

            This input can also be a string pointing to a single SAM
            config file. In this case, the ``config`` column of the
            CSV points input should be set to ``None`` or left out
            completely. See the documentation for the ``reV`` SAM class
            (e.g. :class:`reV.SAM.generation.WindPower`,
            :class:`reV.SAM.generation.PvWattsv8`,
            :class:`reV.SAM.generation.Geothermal`, etc.) for
            documentation on the allowed and/or required SAM config file
            inputs.
        resource_file : str
            Filepath to resource data. This input can be path to a
            single resource HDF5 file, a path to a directory containing
            data spread across multiple HDF5 files, or a path including
            a wildcard input like ``/h5_dir/prefix*suffix``. In all
            cases, the resource data must be readable by
            :py:class:`rex.resource.Resource`
            or :py:class:`rex.multi_file_resource.MultiFileResource`.
            (i.e. the resource data conform to the
            `rex data format <https://tinyurl.com/3fy7v5kx>`_). This
            means the data file(s) must contain a 1D ``time_index``
            dataset indicating the UTC time of observation, a 1D
            ``meta`` dataset represented by a DataFrame with
            site-specific columns, and 2D resource datasets that match
            the dimensions of (``time_index``, ``meta``). The time index
            must start at 00:00 of January 1st of the year under
            consideration, and its shape must be a multiple of 8760.
            If executing ``reV`` from the command line, this path can
            contain brackets ``{}`` that will be filled in by the
            `analysis_years` input.

            .. Important:: If you are using custom resource data (i.e.
              not NSRDB/WTK/Sup3rCC, etc.), ensure the following:

                  - The data conforms to the
                    `rex data format <https://tinyurl.com/3fy7v5kx>`_.
                  - The ``meta`` DataFrame is organized such that every
                    row is a pixel and at least the columns
                    ``latitude``, ``longitude``, ``timezone``, and
                    ``elevation`` are given for each location.
                  - The time index and associated temporal data is in
                    UTC.
                  - The latitude is between -90 and 90 and longitude is
                    between -180 and 180.
                  - For solar data, ensure the DNI/DHI are not zero. You
                    can calculate one of these these inputs from the
                    other using the relationship

                    .. math:: GHI = DNI * cos(SZA) + DHI

        low_res_resource_file : str, optional
            Optional low resolution resource file that will be
            dynamically mapped+interpolated to the nominal-resolution
            `resource_file`. This needs to be of the same format as
            `resource_file` - both files need to be handled by the
            same ``rex Resource`` handler (e.g. ``WindResource``). All
            of the requirements from the `resource_file` apply to this
            input as well. If ``None``, no dynamic mapping to higher
            resolutions is performed. By default, ``None``.
        output_request : list | tuple, optional
            List of output variables requested from SAM. Can be any
            of the parameters in the "Outputs" group of the PySAM module
            (e.g. :py:class:`PySAM.Windpower.Windpower.Outputs`,
            :py:class:`PySAM.Pvwattsv8.Pvwattsv8.Outputs`,
            :py:class:`PySAM.Geothermal.Geothermal.Outputs`, etc.) being
            executed. This list can also include a select number of SAM
            config/resource parameters to include in the output:
            any key in any of the
            `output attribute JSON files <https://tinyurl.com/4bmrpe3j/>`_
            may be requested. If ``cf_mean`` is not included in this
            list, it will automatically be added. Time-series profiles
            requested via this input are output in UTC.

            .. Note:: If you are performing ``reV`` solar runs using
              ``PVWatts`` and would like ``reV`` to include AC capacity
              values in your aggregation/supply curves, then you must
              include the ``"dc_ac_ratio"`` time series as an output in
              `output_request` when running ``reV`` generation. The AC
              capacity outputs will automatically be added during the
              aggregation/supply curve step if the ``"dc_ac_ratio"``
              dataset is detected in the generation file.

            By default, ``('cf_mean',)``.
        site_data : str | pd.DataFrame, optional
            Site-specific input data for SAM calculation. If this input
            is a string, it should be a path that points to a CSV file.
            Otherwise, this input should be a DataFrame with
            pre-extracted site data. Rows in this table should match
            the input sites via a ``gid`` column. The rest of the
            columns should match configuration input keys that will take
            site-specific values. Note that some or all site-specific
            inputs can be specified via the `project_points` input
            table instead. If ``None``, no site-specific data is
            considered. By default, ``None``.
        curtailment : dict | str, optional
            Inputs for curtailment parameters, which can be:

                - Explicit namespace of curtailment variables (dict)
                - Pointer to curtailment config file with path (str)

            The allowed key-value input pairs in the curtailment
            configuration are documented as properties of the
            :class:`reV.config.curtailment.Curtailment` class. If
            ``None``, no curtailment is modeled. By default, ``None``.
        gid_map : dict | str, optional
            Mapping of unique integer generation gids (keys) to single
            integer resource gids (values). This enables unique
            generation gids in the project points to map to non-unique
            resource gids, which can be useful when evaluating multiple
            resource datasets in ``reV`` (e.g., forecasted ECMWF
            resource data to complement historical WTK meteorology).
            This input can be a pre-extracted dictionary or a path to a
            JSON or CSV file. If this input points to a CSV file, the
            file must have the columns ``gid`` (which matches the
            project points) and ``gid_map`` (gids to extract from the
            resource input). If ``None``, the GID values in the project
            points are assumed to match the resource GID values.
            By default, ``None``.
        drop_leap : bool, optional
            Drop leap day instead of final day of year when handling
            leap years. By default, ``False``.
        sites_per_worker : int, optional
            Number of sites to run in series on a worker. ``None``
            defaults to the resource file chunk size.
            By default, ``None``.
        memory_utilization_limit : float, optional
            Memory utilization limit (fractional). Must be a value
            between 0 and 1. This input sets how many site results will
            be stored in-memory at any given time before flushing to
            disk. By default, ``0.4``.
        scale_outputs : bool, optional
            Flag to scale outputs in-place immediately upon ``Gen``
            returning data. By default, ``True``.
        write_mapped_gids : bool, optional
            Option to write mapped gids to output meta instead of
            resource gids. By default, ``False``.
        bias_correct : str | pd.DataFrame, optional
            Optional DataFrame or CSV filepath to a wind or solar
            resource bias correction table. This has columns:

                - ``gid``: GID of site (can be index name)
                - ``adder``: Value to add to resource at each site
                - ``scalar``: Value to scale resource at each site by

            The ``gid`` field should match the true resource ``gid``
            regardless of the optional ``gid_map`` input. If both
            ``adder`` and ``scalar`` are present, the wind or solar
            resource is corrected by :math:`(res*scalar)+adder`. If
            *either* is missing, ``scalar`` defaults to 1 and
            ``adder`` to 0. Only `windspeed` **or** `GHI` + `DNI` are
            corrected, depending on the technology (wind for the former,
            solar for the latter). `GHI` and `DNI` are corrected with
            the same correction factors. If ``None``, no corrections are
            applied. By default, ``None``.
        """
        pc = self.get_pc(points=project_points, points_range=None,
                         sam_configs=sam_files, tech=technology,
                         sites_per_worker=sites_per_worker,
                         res_file=resource_file,
                         curtailment=curtailment)

        super().__init__(pc, output_request, site_data=site_data,
                         drop_leap=drop_leap,
                         memory_utilization_limit=memory_utilization_limit,
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
            pool_size=os.cpu_count() * 2):
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
        pool_size : int, optional
            Number of futures to submit to a single process pool for
            parallel futures. By default, ``os.cpu_count() * 2``.

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
