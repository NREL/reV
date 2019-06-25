"""
reV Supply Curve Points
"""
import os
import h5py
import pandas as pd
import numpy as np
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.handlers.geotiff import Geotiff
from reV.utilities.exceptions import (SupplyCurveError, SupplyCurveInputError,
                                      EmptySupplyCurvePointError)


class ExclusionPoints(Geotiff):
    """Exclusion points framework"""

    def __init__(self, fpath, chunks=(128, 128)):
        """
        Parameters
        ----------
        fpath : str
            Path to .tiff file.
        chunks : tuple
            GeoTIFF chunk (tile) shape/size.
        """

        super().__init__(fpath, chunks=chunks)


class SupplyCurvePoint:
    """Single supply curve point framework"""

    def __init__(self, gid, fpath_excl, fpath_gen, fpath_techmap,
                 resolution=64):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        resolution : int | None
            SC resolution, must be input in combination with gid.
        """

        self._gid = gid
        self._fpath_excl = None
        self._fpath_gen = None
        self._excl_meta = None
        self._gen_ind_global = None
        self._lat_lon_lims = None
        self._gen_tree = None
        self._gen_mask = None

        self._parse_sc_point_def(gid, fpath_excl, resolution)
        self._parse_fpaths(fpath_excl, fpath_gen, fpath_techmap)
        self._init_meta()

    def _parse_sc_point_def(self, gid, fpath_excl, resolution):
        """Parse inputs for the definition of this SC point.

        Parameters
        ----------
        gid : int | None
            gid for supply curve point to analyze.
        fpath_excl : str
            Filepath to exclusions geotiff.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        """

        self._sc = SupplyCurveExtent(fpath_excl, resolution=resolution)
        self._rows, self._cols = self._sc.get_excl_slices(gid)

    def _parse_fpaths(self, fpath_excl, fpath_gen, fpath_techmap):
        """Parse filepath inputs and set to attributes.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        fpath_techmap : str
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework).
        """

        if isinstance(fpath_excl, str):
            self._fpath_excl = fpath_excl
            self._exclusions = ExclusionPoints(fpath_excl)
        else:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs an '
                                        'exclusions file path, but received: '
                                        '{}'.format(type(fpath_excl)))

        if isinstance(fpath_gen, str):
            self._fpath_gen = fpath_gen
            self._gen = Outputs(fpath_gen, str_decode=False)
        else:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs a '
                                        'generation output file path, but '
                                        'received: {}'
                                        .format(type(fpath_gen)))

        if isinstance(fpath_techmap, str):
            self._fpath_techmap = fpath_techmap
            self._techmap = h5py.File(fpath_techmap, 'r')

            map_attrs = dict(self._techmap.attrs)

            if (os.path.basename(fpath_gen) !=
                    os.path.basename(map_attrs['fpath_gen'])):
                warn('Input generation file name ("{}") does not match tech '
                     'file attribute ("{}")'
                     .format(os.path.basename(fpath_gen),
                             os.path.basename(map_attrs['fpath_gen'])))

            if (os.path.basename(fpath_excl) !=
                    os.path.basename(map_attrs['fpath_excl'])):
                warn('Input exclusion file name ("{}") does not match tech '
                     'file attribute ("{}")'
                     .format(os.path.basename(fpath_excl),
                             os.path.basename(map_attrs['fpath_excl'])))

        else:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs a '
                                        'techmap file path, but '
                                        'received: {}'
                                        .format(type(fpath_techmap)))

    def _init_meta(self):
        """Initialize a SC point meta data object from the the tech map."""

        gen_gids = self._techmap['gen_ind'][self._rows, self._cols].flatten()
        valid_points = (gen_gids != -1)
        gen_gids = gen_gids[valid_points]

        if gen_gids.size == 0:
            msg = ('Supply curve point gid {} has no viable exclusion points '
                   'based on exclusion and gen files: "{}", "{}"'
                   .format(self._gid, self._fpath_excl, self._fpath_gen))
            raise EmptySupplyCurvePointError(msg)

        lats = self._techmap['latitude'][self._rows, self._cols].flatten()
        lons = self._techmap['longitude'][self._rows, self._cols].flatten()
        self._centroid = (lats.mean(), lons.mean())

        self._meta = pd.DataFrame(
            {'latitude': lats[valid_points],
             'longitude': lons[valid_points],
             'gen_gid': gen_gids,
             'res_gid': self.gen.meta['gid'].values[gen_gids]})

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def close(self):
        """Close all file handlers."""
        self._exclusions.close()
        self._gen.close()
        self._techmap.close()

    @property
    def meta(self):
        """Get the meta data related to this supply curve point.

        Returns
        -------
        meta : pd.DataFrame
            Meta data for this supply curve point based on the tech exclusion
            points that make up this sc point.
        """
        return self._meta

    @property
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _exclusions : ExclusionPoints
            Exclusions geotiff handler object.
        """
        return self._exclusions

    @property
    def centroid(self):
        """Get the supply curve point centroid coordinate.

        Returns
        -------
        centroid : tuple
            SC point centroid (lat, lon).
        """

        return self._centroid

    @property
    def gen(self):
        """Get the generation output object.

        Returns
        -------
        _gen : Outputs
            reV generation outputs object
        """
        return self._gen

    @property
    def techmap(self):
        """Get the reV technology mapping object.

        Returns
        -------
        _techmap : h5py.File
            reV techmap file object.
        """
        return self._techmap


class SupplyCurveExtent:
    """Supply curve full extent framework."""

    def __init__(self, fpath_excl, resolution=64):
        """
        Parameters
        ----------
        fpath_excl : str | ExclusionPoints
            File path to the exclusions grid, or pre-initialized
            ExclusionPoints. The exclusions dictate the SC analysis extent.
        resolution : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """

        if not isinstance(resolution, int):
            raise SupplyCurveInputError('Supply Curve resolution needs to be '
                                        'an integer but received: {}'
                                        .format(type(resolution)))

        if isinstance(fpath_excl, str):
            self._fpath_excl = fpath_excl
            self._exclusions = ExclusionPoints(fpath_excl)
        else:
            raise SupplyCurveInputError('SupplyCurvePoints needs an '
                                        'exclusions file path, but received: '
                                        '{}'.format(type(fpath_excl)))

        # limit the resolution to the exclusion shape.
        self._res = int(np.min(self.exclusions.shape + (resolution, )))

        self._cols_of_excl = None
        self._rows_of_excl = None
        self._points = None

    def __len__(self):
        """Total number of supply curve points."""
        return self.n_rows * self.n_cols

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def __getitem__(self, gid):
        """Get SC extent meta data corresponding to an SC point gid."""
        if gid >= len(self):
            raise KeyError('SC extent with {} points does not contain SC '
                           'point gid {}.'.format(len(self), gid))
        return self.points.loc[gid]

    def close(self):
        """Close all file handlers."""
        self._exclusions.close()

    @property
    def shape(self):
        """Get the Supply curve shape tuple (n_rows, n_cols).

        Returns
        -------
        shape : tuple
            2-entry tuple representing the full supply curve extent.
        """

        return (self.n_rows, self.n_cols)

    @property
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _exclusions : ExclusionPoints
            Exclusions geotiff handler object.
        """
        return self._exclusions

    @property
    def resolution(self):
        """Get the 1D resolution.

        Returns
        -------
        _res : int
            Number of exclusion points per SC point along an axis.
            This number**2 is the total number of exclusion points per
            SC point.
        """
        return self._res

    @property
    def excl_rows(self):
        """Get the unique row indices identifying the exclusion points.

        Returns
        -------
        excl_rows : np.ndarray
            Array of exclusion row indices.
        """
        return np.arange(self.exclusions.n_rows)

    @property
    def excl_cols(self):
        """Get the unique column indices identifying the exclusion points.

        Returns
        -------
        excl_cols : np.ndarray
            Array of exclusion column indices.
        """
        return np.arange(self.exclusions.n_cols)

    @property
    def rows_of_excl(self):
        """List representing the supply curve points rows and which
        exclusions rows belong to each supply curve row.

        Returns
        -------
        _rows_of_excl : list
            List representing the supply curve points rows. Each list entry
            contains the exclusion row indices that are included in the sc
            point.
        """
        if self._rows_of_excl is None:
            self._rows_of_excl = self._chunk_excl(self.excl_rows)
        return self._rows_of_excl

    @property
    def cols_of_excl(self):
        """List representing the supply curve points columns and which
        exclusions columns belong to each supply curve column.

        Returns
        -------
        _cols_of_excl : list
            List representing the supply curve points columns. Each list entry
            contains the exclusion column indices that are included in the sc
            point.
        """
        if self._cols_of_excl is None:
            self._cols_of_excl = self._chunk_excl(self.excl_cols)
        return self._cols_of_excl

    @property
    def n_rows(self):
        """Get the number of supply curve grid rows.

        Returns
        -------
        n_rows : int
            Number of row entries in the full supply curve grid.
        """
        return int(np.ceil(self.exclusions.n_rows / self.resolution))

    @property
    def n_cols(self):
        """Get the number of supply curve grid columns.

        Returns
        -------
        n_cols : int
            Number of column entries in the full supply curve grid.
        """
        return int(np.ceil(self.exclusions.n_cols / self.resolution))

    @property
    def points(self):
        """Get the summary dataframe of supply curve points.

        Returns
        -------
        _points : pd.DataFrame
            Supply curve points with columns for attributes of each sc point.
        """

        if self._points is None:
            sc_col_ind, sc_row_ind = np.meshgrid(np.arange(self.n_cols),
                                                 np.arange(self.n_rows))
            self._points = pd.DataFrame({'row_ind': sc_row_ind.flatten(),
                                         'col_ind': sc_col_ind.flatten()})
            self._points.index.name = 'gid'
        return self._points

    def get_sc_point_obj(self, gid, fpath_gen, **kwargs):
        """Get the single-supply curve point object for the sc point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        sc_point : SingleSupplyCurvePoint
            Single SC point object corresponding to the gid.
        """
        row_slice, col_slice = self.get_excl_slices(gid)
        sc_point = SupplyCurvePoint(self._fpath_excl, fpath_gen,
                                    excl_row_slice=row_slice,
                                    excl_col_slice=col_slice,
                                    **kwargs)
        return sc_point

    def get_excl_slices(self, gid):
        """Get the row and column slices of the exclusions grid corresponding
        to the supply curve point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        row_slice : int
            Exclusions grid row index slice corresponding to the sc point gid.
        col_slice : int
            Exclusions grid col index slice corresponding to the sc point gid.
        """

        if gid >= len(self):
            raise SupplyCurveError('Requested gid "{}" is out of bounds for '
                                   'supply curve points with length "{}".'
                                   .format(gid, len(self)))

        sc_row_ind = self.points.loc[gid, 'row_ind']
        sc_col_ind = self.points.loc[gid, 'col_ind']
        excl_rows = self.rows_of_excl[sc_row_ind]
        excl_cols = self.cols_of_excl[sc_col_ind]
        row_slice = slice(np.min(excl_rows), np.max(excl_rows) + 1)
        col_slice = slice(np.min(excl_cols), np.max(excl_cols) + 1)
        return row_slice, col_slice

    def get_flat_excl_ind(self, gid):
        """Get the index values of the flattened exclusions grid corresponding
        to the supply curve point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        excl_ind : np.ndarray
            Index values of the flattened exclusions grid corresponding to
            the SC gid.
        """

        row_slice, col_slice = self.get_excl_slices(gid)
        excl_ind = self.exclusions.iarr[row_slice, col_slice].flatten()
        return excl_ind

    def get_excl_points(self, dset, gid):
        """Get the exclusions data corresponding to a supply curve gid.

        Parameters
        ----------
        dset : str | int
            Used as the first arg in the exclusions __getitem__ slice.
            String can be "meta", integer can be layer number.
        gid : int
            Supply curve point gid.

        Returns
        -------
        excl_points : pd.DataFrame
            Exclusions data reduced to just the exclusion points associated
            with the requested supply curve gid.
        """

        row_slice, col_slice = self.get_excl_slices(gid)
        return self.exclusions[dset, row_slice, col_slice]

    def get_coord(self, gid):
        """Get the centroid coordinate for the supply curve gid point.

        Parameters
        ----------
        gid : int
            Supply curve point gid.

        Returns
        -------
        coord : tuple
            Two entry coordinate tuple: (latitude, longitude)
        """

        excl_meta = self.get_excl_points('meta', gid)
        lat = (excl_meta['latitude'].min() + excl_meta['latitude'].max()) / 2
        lon = (excl_meta['longitude'].min() + excl_meta['longitude'].max()) / 2
        return (lat, lon)

    def _chunk_excl(self, arr):
        """Split an array into a list of arrays with len == resolution.

        Parameters
        ----------
        arr : np.ndarray
            1D array to be split into chunks.

        Returns
        -------
        chunks : list
            List of arrays, each with length equal to self.resolution
            (except for the last array in the list which is the remainder).
        """

        chunks = []
        i = 0
        while True:
            if i == len(arr):
                break
            else:
                chunks.append(arr[i:i + self.resolution])
            i = np.min((len(arr), i + self.resolution))

        return chunks
