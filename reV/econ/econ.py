"""
Levelized Cost of Energy
"""
import logging
import pandas as pd
from warnings import warn

from reV.SAM.econ import LCOE as SAM_LCOE
from reV.SAM.econ import SingleOwner
from reV.handlers.outputs import Outputs
from reV.utilities.execution import execute_parallel, execute_single
from reV.generation.generation import Gen


logger = logging.getLogger(__name__)


class Econ(Gen):
    """Base econ class"""

    # Mapping of reV econ output strings to SAM econ functions
    OPTIONS = {'lcoe_fcr': SAM_LCOE.reV_run,
               'ppa_price': SingleOwner.reV_run,
               }

    # Mapping of reV econ outputs to scale factors and units
    OUT_ATTRS = {'lcoe_fcr': {'scale_factor': 1, 'units': 'dol/MWh',
                              'dtype': 'float32', 'chunks': None,
                              'type': 'scalar'},
                 'ppa_price': {'scale_factor': 1, 'units': 'dol/MWh',
                               'dtype': 'float32', 'chunks': None,
                               'type': 'scalar'},
                 }

    def __init__(self, points_control, cf_file, cf_year, site_data=None,
                 output_request='lcoe_fcr', fout=None, dirout='./econ_out',
                 mem_util_lim=0.7):
        """Initialize an econ instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            Project points control instance for site and SAM config spec.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str
            reV generation year to calculate econ for. cf_year='my' will look
            for the multi-year mean generation results.
        site_data : str | pd.DataFrame | None
            Site-specific data for econ calculation. Str points to csv,
            DataFrame is pre-extracted data. Rows match sites, columns are
            variables. Input as None if the only site data required is present
            in the cf_file.
        output_request : str | tuple
            Economic output variable requested from SAM (lcoe_fcr, ppa_price).
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        mem_util_lim : float
            Memory utilization limit (fractional). This sets how many site
            results will be stored in-memory at any given time before flushing
            to disk.
        """

        self._points_control = points_control
        self._cf_file = cf_file
        self._cf_year = cf_year
        self._fout = fout
        self._dirout = dirout
        self.output_request = output_request
        self.mem_util_lim = mem_util_lim
        if site_data:
            self.site_data = site_data

    @Gen.output_request.setter  # pylint: disable-msg=E1101
    def output_request(self, req):
        """Set the single output variable requested from econ.

        Parameters
        ----------
        req : str | tuple
            Single econ output variable requested from SAM.
        """

        if isinstance(req, (list, tuple)) and len(req) == 1:
            # ensure single string output request
            self._output_request = req
        elif isinstance(req, str):
            self._output_request = (req, )
        else:
            raise TypeError('Econ output request must be a single variable '
                            'string or single entry list, but received: {}'
                            .format(req))

        if self._output_request[0] not in self.OPTIONS:
            raise KeyError('Requested econ variable "{}" is not available. '
                           'reV econ can analyze the following: {}'
                           .format(self._output_request,
                                   list(self.OPTIONS.keys())))

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

    @site_data.setter
    def site_data(self, inp):
        """Set the site data attribute

        Parameters
        ----------
        inp : str | pd.DataFrame
            Site data in .csv or pre-extracted dataframe format.
        """

        if isinstance(inp, str):
            if inp.endswith('.csv'):
                self._site_data = pd.read_csv(inp)
        elif isinstance(inp, pd.DataFrame):
            self._site_data = inp

        if not hasattr(self, '_site_data'):
            # site data was not able to be set. Raise error.
            raise Exception('Site data input must be .csv or '
                            'dataframe, but received: {}'.format(inp))

        if ('gid' not in self._site_data and
                self._site_data.index.name != 'gid'):
            # require gid as column label or index
            raise KeyError('Site data input must have "gid" column to match '
                           'reV site index.')

        if self._site_data.index.name != 'gid':
            # make gid index if not already
            self._site_data = self._site_data.\
                set_index('gid', drop=True)

    @property
    def cf_year(self):
        """Get the year to analyze.

        Returns
        -------
        cf_year : int | str
            reV generation year to calculate econ for. cf_year='my' will look
            for the multi-year mean generation results.
        """
        return self._cf_year

    @staticmethod
    def handle_fout(fout, dirout, year):
        """Ensure that the file+dir output exist and have unique names.

        Parameters
        ----------
        fout : str
            Target filename (with or without .h5 extension).
        dirout : str
            Target output directory.
        year : str | int
            Analysis year to be added to the fout.

        Returns
        -------
        fout : str
            Target output directory joined with the target filename.
        """
        # combine filename and path
        fout = Econ.make_h5_fpath(fout, dirout)

        if str(year) not in fout:
            # add year tag to fout
            if fout.endswith('.h5'):
                fout = fout.replace('.h5', '_{}.h5'.format(year))

        return fout

    @property
    def meta(self):
        """Get meta data from the source capacity factors file.

        Returns
        -------
        _meta : pd.DataFrame
            Meta data from capacity factor outputs file.
        """

        if not hasattr(self, '_meta'):
            with Outputs(self.cf_file) as cfh:
                self._meta = cfh.meta
        return self._meta

    @property
    def time_index(self):
        """Get the generation resource time index data."""
        if not hasattr(self, '_time_index'):
            with Outputs(self.cf_file) as cfh:
                if 'time_index' in cfh.dsets:
                    self._time_index = cfh.time_index
                else:
                    self._time_index = None

        return self._time_index

    @property
    def site_df(self):
        """Get the dataframe of site-specific variables.

        Returns
        -------
        _site_df : pd.DataFrame
            Dataframe of site-specific input variables. Number of rows should
            match the number of sites, column labels are the variable keys
            that will be passed forward as SAM parameters.
        """
        if not hasattr(self, '_site_df'):
            site_gids = self.meta['gid']
            with Outputs(self.cf_file) as cfh:
                if 'cf_mean' in cfh.dsets:
                    cf_arr = cfh['cf_mean']
                elif 'cf' in cfh.dsets:
                    cf_arr = cfh['cf']
                elif 'cf_{}'.format(self.cf_year) in cfh.dsets:
                    cf_arr = cfh['cf_{}'.format(self.cf_year)]
                elif 'cf_mean' in self.meta:
                    cf_arr = self.meta['cf_mean']
                else:
                    raise KeyError('Could not find "cf_mean", "cf", or "{}" '
                                   'dataset in {}. Looked in both the h5 '
                                   'datasets and the meta data. The following '
                                   'dsets were available: {}.'
                                   .format('cf_{}'.format(self.cf_year),
                                           self.cf_file, cfh.dsets))

            # set site-specific values in dataframe with
            # columns -> variables, rows -> sites
            self._site_df = pd.DataFrame({'capacity_factor': cf_arr},
                                         index=site_gids)

        # check for offshore flag and add to site_df before returning
        if 'offshore' not in str(self._site_df.columns.values):
            self.check_offshore()

        return self._site_df

    def add_site_df(self):
        """Add the site df (site-specific inputs) to project points dataframe.

        This ensures that only the relevant site's data will be passed through
        to dask workers when points_control is iterated and split.
        """
        self.project_points.join_df(self.site_df, key=self.site_df.index)

    def check_offshore(self):
        """Check if input cf data has offshore flags then add to site_df."""
        # only run this once site_df has been set
        if hasattr(self, '_site_df'):
            # only run this if offshore has not yet been added to site_df
            if 'offshore' not in self._site_df:
                # check for offshore flags for wind data
                if 'offshore' in self.meta:
                    logger.info('Found "offshore" data in meta. Interpreting '
                                'as wind sites that may be analyzed using '
                                'ORCA.')
                    # save offshore flags as boolean array
                    self._site_df['offshore'] = self.meta['offshore']\
                        .astype(bool)

                    # if available, merge site data into site_df
                    if hasattr(self, '_site_data'):
                        self._site_df = pd.merge(self._site_df,
                                                 self.site_data,
                                                 how='left', left_index=True,
                                                 right_index=True,
                                                 suffixes=['', '_site'],
                                                 copy=False, validate='1:1')

    def econ_to_disk(self, fout='econ_out.h5'):
        """Save econ results to disk.

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        """

        # retrieve the dataset with associated attributes
        data, dtype, chunks, attrs = self.get_dset_attrs(
            self.output_request[0])
        # write econ results means to disk
        Outputs.write_means(h5_file=fout, meta=self.meta,
                            dset_name=self.output_request[0], means=data,
                            attrs=attrs, dtype=dtype, chunks=chunks,
                            sam_configs=self.sam_configs)

    def flush(self):
        """Flush econ data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the isntance attributes
        "self.fout" and "self.dirout". The flushed file is ensured to have a
        unique filename. Data is not flushed if fout is None or if .out is
        empty.
        """

        # use mutable copies of the properties
        fout = self.fout
        dirout = self.dirout

        # handle output file request if file is specified and .out is not empty
        if isinstance(fout, str) and self.out:
            fout = self.handle_fout(fout, dirout, self.cf_year)

            logger.info('Flushing econ outputs to disk, target file: {}'
                        .format(fout))
            self.econ_to_disk(fout=fout)
            logger.debug('Flushed econ output successfully to disk.')

    @staticmethod
    def run(pc, output_request):
        """Run the SAM econ calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
            Must have project_points with df property with all relevant
            site-specific inputs and a 'gid' column. By passing site-specific
            inputs in this dataframe, which was split using points_control,
            only the data relevant to the current sites should be passed here.
        """

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
        out = Econ.OPTIONS[output_request](pc, site_df,
                                           output_request=output_request)
        return out

    @classmethod
    def run_direct(cls, points=None, sam_files=None, cf_file=None,
                   cf_year=None, site_data=None, output_request='lcoe_fcr',
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
        cf_year : int | str
            reV generation year to calculate econ for. cf_year='my' will look
            for the multi-year mean generation results.
        site_data : str | pd.DataFrame | None
            Site-specific data for econ calculation. Str points to csv,
            DataFrame is pre-extracted data. Rows match sites, columns are
            variables. Input as None if the only site data required is present
            in the cf_file.
        output_request : str
            Economic output variable requested from SAM (lcoe_fcr, ppa_price).
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
        kwargs = {'output_request': output_request}

        # add site_df to project points dataframe
        econ.add_site_df()

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(econ.run, pc, **kwargs)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(econ.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'], **kwargs)

        # save output data to object attribute
        econ.out = out

        # flush output data (will only write to disk if fout is a str)
        econ.flush()

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return econ
