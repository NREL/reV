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
                                      EmptySupplyCurvePointError,
                                      FileInputWarning)


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

    def __init__(self, gid, f_excl, f_gen, f_techmap, resolution=64,
                 exclusion_shape=None, close=True):
        """
        Parameters
        ----------
        gid : int
            gid for supply curve point to analyze.
        f_excl : str | ExclusionPoints
            Filepath to exclusions geotiff or ExclusionPoints file handler.
        f_gen : str | reV.handlers.Outputs
            Filepath to .h5 reV generation output results or reV Outputs file
            handler.
        f_techmap : str | h5py.File
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework) or an h5py file
            handler object.
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.
        close : bool
            Flag to close object file handlers on exit.
        """

        self._gid = gid
        self._close = close

        # filepaths
        self._fpath_excl = None
        self._fpath_gen = None
        self._fpath_techmap = None

        # handler objects
        self._exclusions = None
        self._gen = None
        self._techmap = None

        # Parse inputs
        self._parse_files_ge(f_excl, f_gen)
        self._parse_files_tm(f_techmap)
        self._rows, self._cols = self._parse_slices(gid, f_excl, resolution,
                                                    exclusion_shape)
        self._gen_gids, self._excl_mask = self._parse_tech_map()
        self._res_gids = self._parse_res_gids()
        self._centroid = self._parse_centroid()

    def _parse_files_ge(self, f_excl, f_gen):
        """Parse gen + excl filepath input or handler object and set to attrs.

        Parameters
        ----------
        f_excl : str | ExclusionPoints
            Filepath to exclusions geotiff or ExclusionPoints file handler
        f_gen : str | reV.handlers.Outputs
            Filepath to .h5 reV generation output results or reV Outputs file
            handler.
        """

        if isinstance(f_excl, str):
            self._fpath_excl = f_excl
        elif isinstance(f_excl, ExclusionPoints):
            self._fpath_excl = f_excl._fpath
            self._exclusions = f_excl
        else:
            raise SupplyCurveInputError('SupplyCurvePoints needs an '
                                        'exclusions file path, or '
                                        'ExclusionPoints handler but '
                                        'received: {}'
                                        .format(type(f_excl)))

        if isinstance(f_gen, str):
            self._fpath_gen = f_gen
        elif isinstance(f_gen, Outputs):
            self._fpath_gen = f_gen._h5_file
            self._gen = f_gen
        else:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs a '
                                        'generation output file path or '
                                        'output handler, but received: {}'
                                        .format(type(f_gen)))

    def _parse_files_tm(self, f_techmap):
        """Parse techmap filepath input or handler object and set to attrs.

        Parameters
        ----------
        f_techmap : str | h5py.File
            Filepath to tech mapping between exclusions and generation results
            (created using the reV TechMapping framework) or an h5py file
            handler object.
        """

        if isinstance(f_techmap, str):
            self._fpath_techmap = f_techmap
        elif isinstance(f_techmap, h5py.File):
            self._fpath_techmap = f_techmap.filename
            self._techmap = f_techmap
        else:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs a '
                                        'techmap file path or h5py file '
                                        'handler, but received: {}'
                                        .format(type(f_techmap)))

        # check file paths against tech map file meta attrs
        if self._fpath_gen is not None:
            tm_fgen = os.path.basename(self.techmap.attrs['fpath_gen'])
            if (os.path.basename(self._fpath_gen) != tm_fgen):
                warn('Input generation file name ("{}") does not match tech '
                     'file attribute ("{}")'
                     .format(os.path.basename(self._fpath_gen), tm_fgen),
                     FileInputWarning)
        if self._fpath_excl is not None:
            tm_fexcl = os.path.basename(self.techmap.attrs['fpath_excl'])
            if (os.path.basename(self._fpath_excl) != tm_fexcl):
                warn('Input exclusion file name ("{}") does not match tech '
                     'file attribute ("{}")'
                     .format(os.path.basename(self._fpath_excl), tm_fexcl),
                     FileInputWarning)

    def _parse_slices(self, gid, f_excl, resolution, exclusion_shape):
        """Parse inputs for the definition of this SC point.

        Parameters
        ----------
        gid : int | None
            gid for supply curve point to analyze.
        f_excl : str | ExclusionPoints
            Filepath to exclusions geotiff or ExclusionPoints file handler
        resolution : int | None
            SC resolution, must be input in combination with gid.
        exclusion_shape : tuple
            Shape of the exclusions extent (rows, cols). Inputing this will
            speed things up considerably.

        Returns
        -------
        rows : slice
            Row slice to index the high-res layer (exclusions) for the gid in
            the agg layer (supply curve).
        cols : slice
            Col slice to index the high-res layer (exclusions) for the gid in
            the agg layer (supply curve).
        """

        if exclusion_shape is None:
            sc = SupplyCurveExtent(f_excl, resolution=resolution)
            rows, cols = sc.get_excl_slices(gid)
        else:
            rows, cols = self.get_agg_slices(gid, exclusion_shape, resolution)

        return rows, cols

    def _parse_tech_map(self):
        """Parse data from the tech map file (exclusions to gen mapping).

        Returns
        -------
        gen_gids : np.ndarray
            reV generation gids from the fpath_gen file corresponding to the
            tech exclusions
        valid_points : np.ndarray
            Boolean mask for valid exclusion points (with corresponding gen
            point).
        """

        gen_gids = self.techmap['gen_ind'][self.rows, self.cols].flatten()
        valid_points = (gen_gids != -1)
        gen_gids = gen_gids[valid_points]

        if gen_gids.size == 0:
            msg = ('Supply curve point gid {} has no viable exclusion points '
                   'based on exclusion and gen files: "{}", "{}"'
                   .format(self._gid, self._fpath_excl, self._fpath_gen))
            raise EmptySupplyCurvePointError(msg)

        return gen_gids, valid_points

    def _parse_res_gids(self):
        """Get the resource gid's based on the generation file and gen gids.

        Returns
        -------
        res_gids : np.ndarray
            reV resource gids from the fpath_gen file corresponding to the
            tech exclusions
        """

        res_gids = self.gen.get_meta_arr('gid')[self._gen_gids]
        return res_gids

    def _parse_centroid(self):
        """Get the SC point centroid from the technology map.

        Returns
        -------
        centroid : tuple
            SC point centroid (lat, lon).
        """

        lats = self.techmap['latitude'][self.rows, self.cols]
        lons = self.techmap['longitude'][self.rows, self.cols]
        centroid = (lats.mean(), lons.mean())
        return centroid

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def close(self):
        """Close all file handlers."""
        if self._close:
            if self._exclusions is not None:
                self._exclusions.close()
            if self._gen is not None:
                self._gen.close()
            if self._techmap is not None:
                self._techmap.close()

    @staticmethod
    def get_agg_slices(gid, shape, resolution):
        """Get the row, col slices of an aggregation gid.

        Parameters
        ----------
        gid : int
            Gid of interest in the aggregated layer.
        shape : tuple
            (row, col) shape tuple of the underlying high-res layer.
        resolution : int
            Resolution of the aggregation: number of pixels in 1D being
            aggregated.

        Returns
        -------
        row_slice : slice
            Row slice to index the high-res layer for the gid in the agg layer.
        col_slice : slice
            Col slice to index the high-res layer for the gid in the agg layer.
        """

        nrows = int(np.ceil(shape[0] / resolution))
        ncols = int(np.ceil(shape[1] / resolution))
        super_shape = (nrows, ncols)
        arr = np.arange(nrows * ncols).reshape(super_shape)
        try:
            loc = np.where(arr == gid)
            row = loc[0][0]
            col = loc[1][0]
        except IndexError as _:
            raise IndexError('Gid {} out of bounds for extent shape {} and '
                             'resolution {}.'.format(gid, shape, resolution))

        if row + 1 != nrows:
            row_slice = slice(row * resolution, (row + 1) * resolution)
        else:
            row_slice = slice(row * resolution, shape[0])

        if col + 1 != ncols:
            col_slice = slice(col * resolution, (col + 1) * resolution)
        else:
            col_slice = slice(col * resolution, shape[1])

        return row_slice, col_slice

    @property
    def rows(self):
        """Get the rows of the exclusions layer associated with this SC point.

        Returns
        -------
        rows : slice
            Row slice to index the high-res layer (exclusions layer) for the
            gid in the agg layer (supply curve layer).
        """
        return self._rows

    @property
    def cols(self):
        """Get the cols of the exclusions layer associated with this SC point.

        Returns
        -------
        cols : slice
            Column slice to index the high-res layer (exclusions layer) for the
            gid in the agg layer (supply curve layer).
        """
        return self._cols

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
    def exclusions(self):
        """Get the exclusions object.

        Returns
        -------
        _exclusions : ExclusionPoints
            Exclusions geotiff handler object.
        """
        if self._exclusions is None:
            self._exclusions = ExclusionPoints(self._fpath_excl)
        return self._exclusions

    @property
    def gen(self):
        """Get the generation output object.

        Returns
        -------
        _gen : Outputs
            reV generation outputs object
        """
        if self._gen is None:
            self._gen = Outputs(self._fpath_gen, str_decode=False)
        return self._gen

    @property
    def techmap(self):
        """Get the reV technology mapping object.

        Returns
        -------
        _techmap : h5py.File
            reV techmap file object.
        """
        if self._techmap is None:
            self._techmap = h5py.File(self._fpath_techmap, 'r')
        return self._techmap


class SupplyCurveExtent:
    """Supply curve full extent framework."""

    def __init__(self, f_excl, resolution=64):
        """
        Parameters
        ----------
        f_excl : str | ExclusionPoints
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

        if isinstance(f_excl, str):
            self._fpath_excl = f_excl
            self._exclusions = ExclusionPoints(f_excl)
        elif isinstance(f_excl, ExclusionPoints):
            self._fpath_excl = f_excl._fpath
            self._exclusions = f_excl
        else:
            raise SupplyCurveInputError('SupplyCurvePoints needs an '
                                        'exclusions file path, or '
                                        'ExclusionPoints handler but '
                                        'received: {}'
                                        .format(type(f_excl)))

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
            self._rows_of_excl = self._chunk_excl(self.excl_rows,
                                                  self.resolution)
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
            self._cols_of_excl = self._chunk_excl(self.excl_cols,
                                                  self.resolution)
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

    def get_sc_point_obj(self, gid, f_gen, **kwargs):
        """Get the single-supply curve point object for the sc point gid.

        Parameters
        ----------
        gid : int
            Supply curve point gid.
        f_gen : str | reV.handlers.Outputs
            Filepath to .h5 reV generation output results or reV Outputs file
            handler.

        Returns
        -------
        sc_point : SingleSupplyCurvePoint
            Single SC point object corresponding to the gid.
        """
        row_slice, col_slice = self.get_excl_slices(gid)
        sc_point = SupplyCurvePoint(self._exclusions, f_gen,
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

    @staticmethod
    def _chunk_excl(arr, resolution):
        """Split an array into a list of arrays with len == resolution.

        Parameters
        ----------
        arr : np.ndarray
            1D array to be split into chunks.
        resolution : int
            Resolution of the chunks.

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
                chunks.append(arr[i:i + resolution])
            i = np.min((len(arr), i + resolution))

        return chunks
