"""
Levelized Cost of Energy
"""
import logging
import pandas as pd
import pprint
from warnings import warn

from reV.SAM.econ import LCOE as SAM_LCOE
from reV.SAM.econ import SingleOwner
from reV.handlers.outputs import Outputs
from reV.utilities.execution import (execute_parallel, execute_single,
                                     SmartParallelJob)
from reV.generation.generation import Gen


logger = logging.getLogger(__name__)


class Econ(Gen):
    """Base econ class"""

    # Mapping of reV econ output strings to SAM econ functions
    OPTIONS = {'lcoe_fcr': SAM_LCOE.reV_run,
               'ppa_price': SingleOwner.reV_run,
               'npv': SingleOwner.reV_run,
               'lcoe_real': SingleOwner.reV_run,
               'lcoe_nom': SingleOwner.reV_run,
               }

    # Mapping of reV econ outputs to scale factors and units.
    # Type is scalar or array and corresponds to the SAM single-site output
    OUT_ATTRS = {'lcoe_fcr': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 'ppa_price': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'npv': {'scale_factor': 1, 'units': 'dol',
                         'dtype': 'float32', 'chunks': None,
                         'type': 'scalar'},
                 'lcoe_real': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 'lcoe_nom': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 }

    def __init__(self, points_control, cf_file, cf_year, site_data=None,
                 output_request='lcoe_fcr', fout=None, dirout='./econ_out',
                 mem_util_lim=0.4):
        """Initialize an econ instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            Project points control instance for site and SAM config spec.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        site_data : str | pd.DataFrame | None
            Site-specific data for econ calculation. Str points to csv,
            DataFrame is pre-extracted data. Rows match sites, columns are
            variables. Input as None if the only site data required is present
            in the cf_file.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        """
        self._points_control = points_control
        self._cf_file = cf_file
        self._year = cf_year
        self._site_limit = None
        self._site_mem = None
        self._fout = fout
        self._dirout = dirout
        self._fpath = None
        self._time_index = None
        self._meta = None
        self.mem_util_lim = mem_util_lim
        self._output_request = self._parse_output_request(output_request)
        self._site_data = self._parse_site_data(site_data)

        # pre-initialize output arrays to store results when available.
        self._out = {}
        self._finished_sites = []
        self._out_n_sites = 0
        self._out_chunk = ()
        self._init_out_arrays()

        # initialize output file
        self._init_fpath()
        self._init_h5()

    def _parse_output_request(self, req):
        """Set the output variables requested from generation.

        Parameters
        ----------
        req : str| list | tuple
            Output variables requested from SAM.

        Returns
        -------
        output_request : tuple
            Output variables requested from SAM.
        """

        if isinstance(req, str):
            # single output request, make tuple
            output_request = (req,)
        elif isinstance(req, (list, tuple)):
            # ensure output request is tuple
            output_request = tuple(req)
        else:
            raise TypeError('Output request must be str, list, or tuple but '
                            'received: {}'.format(type(req)))

        for request in output_request:
            if request not in self.OUT_ATTRS:
                raise ValueError('User output request "{}" not recognized. '
                                 'The following output requests are available '
                                 'in "{}": "{}"'
                                 .format(request, self.__class__,
                                         list(self.OUT_ATTRS.keys())))

            if self.OPTIONS[request] != self.OPTIONS[output_request[0]]:
                msg = ('Econ outputs requested from different SAM modules not '
                       'currently supported. Output request variables "{}" '
                       'and "{}" require SAM modules "{}" and "{}".'
                       .format(request, output_request[0],
                               self.OPTIONS[request],
                               self.OPTIONS[output_request[0]]))
                raise ValueError(msg)

        return output_request

    def _parse_site_data(self, inp):
        """Parse site-specific data from input arg

        Parameters
        ----------
        inp : str | pd.DataFrame | None
            Site data in .csv or pre-extracted dataframe format. None signifies
            that there is no extra site-specific data and that everything will
            be taken from the cf_file (generation outputs).

        Returns
        -------
        site_data : pd.DataFrame
            Site-specific data for econ calculation. Rows correspond to sites,
            columns are variables.
        """

        if not inp:
            # no input, just initialize dataframe with site gids as index
            site_data = pd.DataFrame(index=self.project_points.sites)
        else:
            # explicit input, initialize df
            if isinstance(inp, str):
                if inp.endswith('.csv'):
                    site_data = pd.read_csv(inp)
            elif isinstance(inp, pd.DataFrame):
                site_data = inp
            else:
                # site data was not able to be set. Raise error.
                raise Exception('Site data input must be .csv or '
                                'dataframe, but received: {}'.format(inp))

            if 'gid' not in site_data and site_data.index.name != 'gid':
                # require gid as column label or index
                raise KeyError('Site data input must have "gid" column '
                               'to match reV site gid.')

            if site_data.index.name != 'gid':
                # make gid the dataframe index if not already
                site_data = site_data.set_index('gid', drop=True)

        # add offshore if necessary
        if 'offshore' in site_data:
            # offshore is already in site data df, just make sure it's boolean
            site_data['offshore'] = site_data['offshore'].astype(bool)

        else:
            # offshore not yet in site data df, check to see if in meta
            if 'offshore' in self.meta:
                logger.debug('Found "offshore" data in meta. Interpreting '
                             'as wind sites that may be analyzed using '
                             'ORCA.')
                # save offshore flags as boolean array
                site_data['offshore'] = self.meta['offshore'].astype(bool)

        return site_data

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
    def site_data(self):
        """Get the site-specific dataframe.

        Returns
        -------
        _site_data : pd.DataFrame
            Site-specific data for econ calculation. Rows match sites,
            columns are variables.
        """
        return self._site_data

    def add_site_data_to_pp(self):
        """Add the site df (site-specific inputs) to project points dataframe.

        This ensures that only the relevant site's data will be passed through
        to dask workers when points_control is iterated and split.
        """
        self.project_points.join_df(self.site_data, key=self.site_data.index)

    @property
    def meta(self):
        """Get meta data from the source capacity factors file.

        Returns
        -------
        _meta : pd.DataFrame
            Meta data from capacity factor outputs file.
        """
        if self._meta is None:
            with Outputs(self.cf_file) as cfh:
                # only take meta that belongs to this project's site list
                self._meta = cfh.meta[
                    cfh.meta['gid'].isin(self.points_control.sites)]

            logger.debug('Meta shape is {}'.format(self._meta.shape))

        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data."""
        if self._time_index is None:
            with Outputs(self.cf_file) as cfh:
                if 'time_index' in cfh.dsets:
                    self._time_index = cfh.time_index

        return self._time_index

    @staticmethod
    def run(pc, output_request, **kwargs):
        """Run the SAM econ calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
            Must have project_points with df property with all relevant
            site-specific inputs and a 'gid' column. By passing site-specific
            inputs in this dataframe, which was split using points_control,
            only the data relevant to the current sites is passed.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        kwargs : dict
            Additional input parameters for the SAM run module.
        """

        # make sure output request is a tuple
        if isinstance(output_request, str):
            output_request = (output_request,)

        # Extract the site df from the project points df.
        site_df = pc.project_points.df

        # check that there is a gid column
        if 'gid' not in site_df:
            warn('Econ input "site_df" (in project_points.df) does not have '
                 'a label corresponding to site gid. This may cause an '
                 'incorrect interpretation of site id.')
        else:
            # extract site df from project points df and set gid as index
            site_df = site_df.set_index('gid', drop=True)

        # SAM execute econ analysis based on output request
        out = Econ.OPTIONS[output_request[0]](pc, site_df,
                                              output_request=output_request,
                                              **kwargs)
        return out

    @classmethod
    def run_direct(cls, points=None, sam_files=None, cf_file=None,
                   cf_year=None, site_data=None, output_request=('lcoe_fcr',),
                   n_workers=1, sites_per_split=100, points_range=None,
                   fout=None, dirout='./econ_out', return_obj=True):
        """Execute a econ run directly from source files without config.

        Parameters
        ----------
        points : slice | str | reV.config.project_points.PointsControl
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
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        site_data : str | pd.DataFrame | None
            Site-specific data for econ calculation. Str points to csv,
            DataFrame is pre-extracted data. Rows match sites, columns are
            variables. Input as None if the only site data required is present
            in the cf_file.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        return_obj : bool
            Option to return the Gen object instance.

        Returns
        -------
        econ : reV.econ.Econ
            Econ object instance with outputs stored in .out attribute.
            Only returned if return_obj is True.
        """

        # get a points control instance
        pc = cls.get_pc(points, points_range, sam_files, tech=None)

        # make a Gen class instance to operate with
        econ = cls(pc, cf_file, cf_year=cf_year, site_data=site_data,
                   output_request=output_request, fout=fout, dirout=dirout)

        diff = set(pc.sites) - set(econ.meta['gid'].values)
        if diff:
            raise Exception('The following analysis sites were requested '
                            'through project points for econ but are not '
                            'found in the CF file ("{}"): {}'
                            .format(econ.cf_file, diff))

        # make a kwarg dict
        kwargs = {'output_request': output_request,
                  'cf_file': cf_file,
                  'cf_year': cf_year}

        # add site_df to project points dataframe
        econ.add_site_data_to_pp()

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(econ.run, pc, **kwargs)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(econ.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.econ',
                                            'reV.generation', 'reV.SAM',
                                            'reV.utilities'], **kwargs)

        # save output data to object attribute
        econ.out = out

        # flush output data (will only write to disk if fout is a str)
        econ.flush()

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return econ

    @classmethod
    def run_smart(cls, points=None, sam_files=None, cf_file=None,
                  cf_year=None, site_data=None, output_request=('lcoe_fcr',),
                  n_workers=1, sites_per_split=100, points_range=None,
                  fout=None, dirout='./econ_out'):
        """Execute a econ run directly from source files without config.

        Parameters
        ----------
        points : slice | str | reV.config.project_points.PointsControl
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
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
        site_data : str | pd.DataFrame | None
            Site-specific data for econ calculation. Str points to csv,
            DataFrame is pre-extracted data. Rows match sites, columns are
            variables. Input as None if the only site data required is present
            in the cf_file.
        output_request : str | list | tuple
            Economic output variable(s) requested from SAM.
        n_workers : int
            Number of local workers to run on.
        sites_per_split : int
            Number of sites to run in series on a core.
        points_range : list | None
            Optional two-entry list specifying the index range of the sites to
            analyze. To be taken from the reV.config.PointsControl.split_range
            property.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.

        Returns
        -------
        econ : reV.econ.Econ
            Econ object instance with outputs stored in .out attribute.
            Only returned if return_obj is True.
        """

        # get a points control instance
        pc = cls.get_pc(points, points_range, sam_files, tech=None)

        # make a Gen class instance to operate with
        econ = cls(pc, cf_file, cf_year=cf_year, site_data=site_data,
                   output_request=output_request, fout=fout, dirout=dirout)

        diff = set(pc.sites) - set(econ.meta['gid'].values)
        if diff:
            raise Exception('The following analysis sites were requested '
                            'through project points for econ but are not '
                            'found in the CF file ("{}"): {}'
                            .format(econ.cf_file, diff))

        # make a kwarg dict
        kwargs = {'output_request': output_request,
                  'cf_file': cf_file,
                  'cf_year': cf_year}

        # add site_df to project points dataframe
        econ.add_site_data_to_pp()

        logger.info('Running parallel econ with smart data flushing '
                    'for: {}'.format(pc))
        logger.debug('The following project points were specified: "{}"'
                     .format(points))
        logger.debug('The following SAM configs are available to this run:\n{}'
                     .format(pprint.pformat(sam_files, indent=4)))
        logger.debug('The SAM output variables have been requested:\n{}'
                     .format(output_request))

        try:
            # use SmartParallelJob to manage runs, but set mem limit to 1
            # because Econ() will manage the sites in-memory
            SmartParallelJob.execute(econ, pc, n_workers=n_workers,
                                     mem_util_lim=1.0, **kwargs)
        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed.')
            raise e
