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
                 output_request='lcoe_fcr', fout=None, dirout='./econ_out'):
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
        self._cf_year = cf_year
        self._fout = fout
        self._dirout = dirout
        self.output_request = output_request
        self.site_data = site_data

        # set memory utilization limit as static (not as important as for gen)
        self.mem_util_lim = 0.7

    @Gen.output_request.setter  # pylint: disable-msg=E1101
    def output_request(self, req):
        """Set the output variables requested from econ.

        Parameters
        ----------
        req : str | list | tuple
            Output variable(s) requested from SAM.
        """

        if isinstance(req, str):
            # single output request, make tuple
            self._output_request = (req,)
        elif isinstance(req, (list, tuple)):
            # ensure output request is tuple
            self._output_request = tuple(req)
        else:
            raise TypeError('Output request must be str, list, or tuple but '
                            'received: {}'.format(type(req)))

        for request in self._output_request:
            if request not in self.OUT_ATTRS:
                raise ValueError('User output request "{}" not recognized. '
                                 'The following output requests are available '
                                 'in "{}": "{}"'
                                 .format(request, self.__class__,
                                         list(self.OUT_ATTRS.keys())))
            if self.OPTIONS[request] != self.OPTIONS[self._output_request[0]]:
                msg = ('Econ outputs requested from different SAM modules not '
                       'currently supported. Output request variables "{}" '
                       'and "{}" require SAM modules "{}" and "{}".'
                       .format(request, self._output_request[0],
                               self.OPTIONS[request],
                               self.OPTIONS[self._output_request[0]]))
                raise ValueError(msg)

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
        inp : str | pd.DataFrame | None
            Site data in .csv or pre-extracted dataframe format. None signifies
            that everything will be taken from the cf_file (generation outputs)
        """

        if not inp:
            # no input, just initialize dataframe with site gids as index
            self._site_data = pd.DataFrame(index=self.project_points.sites)
        else:
            # explicit input, initialize df
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
                raise KeyError('Site data input must have "gid" column '
                               'to match reV site index.')

            if self._site_data.index.name != 'gid':
                # make gid index if not already
                self._site_data = self._site_data.\
                    set_index('gid', drop=True)

        # add offshore if necessary
        self.check_offshore()

    def add_site_data_to_pp(self):
        """Add the site df (site-specific inputs) to project points dataframe.

        This ensures that only the relevant site's data will be passed through
        to dask workers when points_control is iterated and split.
        """
        self.project_points.join_df(self.site_data, key=self.site_data.index)

    def check_offshore(self):
        """Ensure offshore boolean flag in site data df if available."""
        # only run this once site_df has been set
        if hasattr(self, '_site_data'):

            if 'offshore' in self._site_data:
                # offshore is already in site data df, just make sure boolean
                self._site_data['offshore'] = self._site_data['offshore']\
                    .astype(bool)

            else:
                # offshore not yet in site data df, check to see if in meta
                if 'offshore' in self.meta:
                    logger.info('Found "offshore" data in meta. Interpreting '
                                'as wind sites that may be analyzed using '
                                'ORCA.')
                    # save offshore flags as boolean array
                    self._site_data['offshore'] = self.meta['offshore']\
                        .astype(bool)

    @property
    def cf_year(self):
        """Get the year to analyze.

        Returns
        -------
        cf_year : int | str | None
            reV generation year to calculate econ for. Looks for cf_mean_{year}
            or cf_profile_{year}. None will default to a non-year-specific cf
            dataset (cf_mean, cf_profile).
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
                # only take meta that belongs to this project's site list
                self._meta = cfh.meta[
                    cfh.meta['gid'].isin(self.points_control.sites)]
            logger.debug('Meta shape is {}'.format(self._meta.shape))
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

    def econ_to_disk(self, fout='econ_out.h5'):
        """Save econ results to disk.

        Parameters
        ----------
        fout : str
            Target .h5 output file (with path).
        """

        with Outputs(fout, mode='w-') as f:
            # Save meta
            f['meta'] = self.meta
            logger.debug("\t- 'meta' saved to disc")

            if self.sam_configs is not None:
                f.set_configs(self.sam_configs)
                logger.debug("\t- SAM configurations saved as attributes "
                             "on 'meta'")

            # iterate through all output requests writing each as a dataset
            for dset in self.output_request:
                # retrieve the dataset with associated attributes
                data, dtype, chunks, attrs = self.get_dset_attrs(dset)
                # Write output dataset to disk
                f._add_dset(dset_name=dset, data=data, dtype=dtype,
                            chunks=chunks, attrs=attrs)

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
    def run(pc, output_request, **kwargs):
        """Run the SAM econ calculation.

        Parameters
        ----------
        pc : reV.config.project_points.PointsControl
            Iterable points control object from reV config module.
            Must have project_points with df property with all relevant
            site-specific inputs and a 'gid' column. By passing site-specific
            inputs in this dataframe, which was split using points_control,
            only the data relevant to the current sites should be passed here.
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
                                     loggers=[__name__, 'reV.econ',
                                              'reV.generation', 'reV.SAM',
                                              'reV.utilities'],
                                     mem_util_lim=1.0, **kwargs)
        except Exception as e:
            logger.exception('SmartParallelJob.execute() failed.')
            raise e
