"""
Levelized Cost of Energy
"""
import logging
import pandas as pd

from reV.SAM.SAM import LCOE as SAM_LCOE
from reV.handlers.outputs import Outputs
from reV.utilities.execution import execute_parallel, execute_single
from reV.generation.generation import Gen


logger = logging.getLogger(__name__)


class LCOE(Gen):
    """Base LCOE class"""

    def __init__(self, points_control, cf_file, orca_file=None, cf_year=2012,
                 output_request=('lcoe_fcr',), fout=None, dirout='./lcoe_out'):
        """Initialize an LCOE instance.

        Parameters
        ----------
        points_control : reV.config.PointsControl
            Project points control instance for site and SAM config spec.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str
            reV generation year to calculate LCOE for. cf_year='my' will look
            for the multi-year mean generation results.
        output_request : list | tuple
            Output variables requested from SAM.
        fout : str | None
            Optional .h5 output file specification.
        dirout : str | None
            Optional output directory specification. The directory will be
            created if it does not already exist.
        """

        self._points_control = points_control
        self._cf_file = cf_file
        self._orca_file = orca_file
        self._cf_year = cf_year
        self._output_request = output_request
        self._fout = fout
        self._dirout = dirout

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
    def orca_file(self):
        """Get the ORCA data filename and path."""
        return self._orca_file

    @property
    def cf_year(self):
        """Get the year to analyze.

        Returns
        -------
        cf_year : int | str
            reV generation year to calculate LCOE for. cf_year='my' will look
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
        fout = LCOE.make_h5_fpath(fout, dirout)

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
                if 'cf_{}'.format(self.cf_year) in str(list(cfh.dsets)):
                    cf_arr = cfh['cf_{}'.format(self.cf_year)]
                elif 'cf_means' in str(list(cfh.dsets)):
                    cf_arr = cfh['cf_means']
                elif 'cf_means' in self.meta:
                    cf_arr = self.meta['cf_means']
                else:
                    raise KeyError('Could not find "cf_means" or "{}" dataset '
                                   'in {}. Looked in both the h5 datasets and '
                                   'the meta data.'
                                   .format('cf_{}'.format(self.cf_year),
                                           self.cf_file))

            # set site-specific values in dataframe with
            # columns -> variables, rows -> sites
            self._site_df = pd.DataFrame({'capacity_factor': cf_arr},
                                         index=site_gids)

        # check for offshore flag and add to site_df before returning
        if 'offshore' not in str(self._site_df.columns.values):
            self.check_offshore()

        return self._site_df

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
                    if hasattr(self, '_orca_file'):
                        if self.orca_file.endswith('.csv'):
                            orca_data = pd.read_csv(self.orca_file)
                            if 'gid' not in orca_data:
                                raise KeyError('ORCA input data must have '
                                               '"gid" column to match reV '
                                               'site index.')
                            orca_data = orca_data.set_index('gid', drop=True)
                        self._site_df = pd.merge(self._site_df, orca_data,
                                                 how='left', left_index=True,
                                                 right_index=True,
                                                 suffixes=['', '_orca'],
                                                 copy=False, validate='1:1')

    def lcoe_to_disk(self, fout='lcoe_out.h5', mode='w'):
        """Save LCOE results to disk.

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        mode : str
            .h5 file write mode (e.g. 'w', 'w-', 'a').
        """
        lcoe_arr = self.unpack_scalars(self.out, sam_var='lcoe_fcr')
        # write means to disk using CapacityFactor class
        attrs = {'scale_factor': 1, 'units': 'dol/MWh'}
        Outputs.write_means(fout, self.meta, 'lcoe', lcoe_arr, attrs,
                            'float32', self.sam_configs, **{'mode': mode})

    def flush(self, mode='w'):
        """Flush LCOE data in self.out attribute to disk in .h5 format.

        The data to be flushed is accessed from the instance attribute
        "self.out". The disk target is based on the isntance attributes
        "self.fout" and "self.dirout". The flushed file is ensured to have a
        unique filename. Data is not flushed if fout is None or if .out is
        empty.

        Parameters
        ----------
        mode : str
            .h5 file write mode (e.g. 'w', 'w-', 'a').
        """

        # use mutable copies of the properties
        fout = self.fout
        dirout = self.dirout

        # handle output file request if file is specified and .out is not empty
        if isinstance(fout, str) and self.out:
            fout = self.handle_fout(fout, dirout, self.cf_year)

            logger.info('Flushing LCOE outputs to disk, target file: {}'
                        .format(fout))
            self.lcoe_to_disk(fout=fout, mode=mode)
            logger.debug('Flushed LCOE output successfully to disk.')

    @staticmethod
    def run(pc, site_df):
        """Run the SAM LCOE calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
        site_df : pd.DataFrame
            Dataframe of site-specific input variables. Number of rows should
            match the number of sites, column labels are the variable keys
            that will be passed forward as SAM parameters.
        """
        site_df = site_df[site_df.index.isin(pc.sites)]
        out = SAM_LCOE.reV_run(pc, site_df)
        return out

    @classmethod
    def run_direct(cls, points=None, sam_files=None, cf_file=None,
                   orca_file=None,
                   cf_year=None, n_workers=1, sites_per_split=100,
                   points_range=None, fout=None, dirout='./lcoe_out',
                   return_obj=True):
        """Execute a generation run directly from source files without config.

        Parameters
        ----------
        points : slice | str | reV.config.project_points.PointsControl
            Slice specifying project points, or string pointing to a project
            points csv, or a fully instantiated PointsControl object.
        sam_files : dict | str | list
            Dict contains SAM input configuration ID(s) and file path(s).
            Keys are the SAM config ID(s), top level value is the SAM path.
            Can also be a single config file str. If it's a list, it is mapped
            to the sorted list of unique configs requested by points csv.
        cf_file : str
            reV generation capacity factor output file with path.
        cf_year : int | str
            reV generation year to calculate LCOE for. 'my' will look for the
            multi-year mean generation results.
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
        lcoe : reV.lcoe.LCOE
            LCOE object instance with outputs stored in .out attribute.
            Only returned if return_obj is True.
        """

        # get a points control instance
        pc = LCOE.get_pc(points, points_range, sam_files, tech=None)

        # make a Gen class instance to operate with
        lcoe = cls(pc, cf_file, orca_file=orca_file, cf_year=cf_year,
                   fout=fout, dirout=dirout)

        diff = set(pc.sites) - set(lcoe.meta['gid'].values)
        if diff:
            raise Exception('The following analysis sites were requested '
                            'through project points for LCOE but are not '
                            'found in the CF file ("{}"): {}'
                            .format(lcoe.cf_file, diff))

        # make a kwarg dict
        kwargs = {'site_df': lcoe.site_df}

        # use serial or parallel execution control based on n_workers
        if n_workers == 1:
            logger.debug('Running serial generation for: {}'.format(pc))
            out = execute_single(lcoe.run, pc, **kwargs)
        else:
            logger.debug('Running parallel generation for: {}'.format(pc))
            out = execute_parallel(lcoe.run, pc, n_workers=n_workers,
                                   loggers=[__name__, 'reV.SAM'], **kwargs)

        # save output data to object attribute
        lcoe.out = out

        # flush output data (will only write to disk if fout is a str)
        lcoe.flush()

        # optionally return Gen object (useful for debugging and hacking)
        if return_obj:
            return lcoe
