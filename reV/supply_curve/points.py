"""
reV Supply Curve Points
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from warnings import warn

from reV.handlers.outputs import Outputs
from reV.handlers.geotiff import Geotiff
from reV.utilities.exceptions import (SupplyCurveError, SupplyCurveInputError,
                                      OutputWarning,
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

    def __init__(self, fpath_excl, fpath_gen, excl_row_slice=None,
                 excl_col_slice=None, gid=None, resolution=None,
                 gen_tree=None, gen_mask=None, distance_upper_bound=0.03):
        """
        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
        excl_row_slice/excl_col_slice : slice | None
            Exclusions row/column slice belonging to this supply curve point.
            Prefered point definition option over gid+resolution.
        gid : int | None
            gid for supply curve point to analyze. Prefered option is to use
            the row/col slices to define the SC point instead.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        gen_tree : cKDTree
            Pre-initialized cKDTree of the generation meta data. None will
            cause this object to make its own cKDTree.
        gen_mask : pd.Series
            Boolean series mask on generation data that was used to create
            the cKDTree. Required if cKDTree is input.
        distance_upper_bound : float | None
            Upper boundary distance for KNN lookup between exclusion points and
            generation points. None will calculate a good distance based on the
            generation meta data coordinates. 0.03 is a good value for a 4km
            resource grid and finer.
        """

        self._fpath_excl = None
        self._fpath_gen = None
        self._excl_meta = None
        self._gen_ind_global = None
        self._lat_lon_lims = None
        self._gen_tree = None
        self._gen_mask = None
        self._distance_upper_bound = distance_upper_bound

        self._parse_sc_point_def(fpath_excl, excl_row_slice, excl_col_slice,
                                 gid, resolution)
        self._parse_fpaths(fpath_excl, fpath_gen)
        self._parse_gen_tree_mask(gen_tree, gen_mask)

        # set the base exlcusions meta, query the NN to gen points, reduce
        # the exclusions meta and add data about gen NN
        self._excl_meta = self.get_base_excl_meta()
        self._excl_mask, self._gen_ind_global = self._query_excl_nn()
        self._reduce_excl_meta()

    def _parse_sc_point_def(self, fpath_excl, excl_row_slice, excl_col_slice,
                            gid, resolution):
        """Parse inputs for the definition of this SC point.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        excl_row_slice/excl_col_slice : slice | None
            Exclusions row/column slice belonging to this supply curve point.
            Prefered point definition option over gid+resolution.
        gid : int | None
            gid for supply curve point to analyze. Prefered option is to use
            the row/col slices to define the SC point instead.
        resolution : int | None
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead.
        """

        if excl_row_slice is not None and excl_col_slice is not None:
            self._excl_row_slice = excl_row_slice
            self._excl_col_slice = excl_col_slice

        elif gid is None and excl_row_slice is None and excl_col_slice is None:
            raise SupplyCurveInputError('SingleSupplyCurvePoint needs either '
                                        'a gid or a row/col slice input but '
                                        'received None.')

        elif (gid is not None and resolution is not None and
              excl_row_slice is None and excl_col_slice is None):
            s = SupplyCurveExtent(fpath_excl, resolution=resolution)
            self._excl_row_slice, self._excl_col_slice = s.get_excl_slices(gid)

        else:
            raise SupplyCurveInputError('Cannot parse the combination of gid, '
                                        'resolution and exclusions row/col '
                                        'slices.')

    def _parse_fpaths(self, fpath_excl, fpath_gen):
        """Parse filepath inputs and set to attributes.

        Parameters
        ----------
        fpath_excl : str
            Filepath to exclusions geotiff.
        fpath_gen : str
            Filepath to .h5 reV generation output results.
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

    def _parse_gen_tree_mask(self, gen_tree, gen_mask):
        """Verify generation tree/mask and set to attributes.

        Parameters
        ----------
        gen_tree : cKDTree
            Pre-initialized cKDTree of the generation meta data. None will
            cause this object to make its own cKDTree.
        gen_mask : pd.Series
            Boolean series mask on generation data that was used to create
            the cKDTree. Required if cKDTree is input.
        """

        if gen_tree is not None and gen_mask is None:
            raise SupplyCurveInputError('SingleSupplyCurvePoint received a '
                                        'pre-built gen ckdtree but no '
                                        'corresponding gen mask.')
        elif gen_tree is not None and gen_mask is not None:
            if len(gen_tree.indices) != gen_mask.sum():
                msg = ('The {} indices in the gen_tree does not correspond to '
                       'the {} True values in the gen_mask. The gen_mask must '
                       'correspond to the tree that was created!'
                       .format(len(gen_tree.indices), gen_mask.sum()))
                raise SupplyCurveInputError(msg)

        self._gen_tree = gen_tree
        self._gen_mask = gen_mask

    def _query_excl_nn(self):
        """Run the tree query and get the exclusions mask and gen indices.

        Returns
        -------
        excl_mask : np.ndarray
            Boolean 1D array showing which exclusion points have valid
            corresponding generation points.
        gen_ind_global : np.ndarray
            Global generation index values for gen points corresponding to the
            valid exclusion points
        """

        dist, ind = self.gen_tree.query(
            self.exclusion_meta[['latitude', 'longitude']])

        excl_mask = (dist < self.distance_upper_bound)
        ind = ind[excl_mask]

        # pylint: disable-msg=C0121
        gen_ind_global = self.gen_mask[(self.gen_mask == True)]\
            .iloc[ind].index.values  # noqa: E712

        gen_ind_global[(ind < 0)] = -1

        return excl_mask, gen_ind_global

    def _reduce_excl_meta(self):
        """Reduce the exclusions meta data to just those points which have
        corresponding generation points, and add a column showing the
        corresponding generation and resource points.
        """

        self._excl_meta = self._excl_meta[self._excl_mask]
        self._excl_meta['gen_gid'] = self._gen_ind_global

        if 'gid' in self.gen.meta:
            self._excl_meta['resource_gid'] = self.gen.meta.loc[
                self._gen_ind_global, 'gid'].values
        else:
            warn('Generation output file does not have resource "gid" field: '
                 '"{}"'.format(self._fpath_gen), OutputWarning)

        if self._excl_meta.empty:
            msg = ('Supply curve point with row/col slices {}/{} at centroid '
                   'lat/lon {} has no viable exclusion points based on '
                   'exclusion and gen files: "{}", "{}"'
                   .format(self._excl_row_slice, self._excl_col_slice,
                           self.centroid, self._fpath_excl, self._fpath_gen))
            raise EmptySupplyCurvePointError(msg)

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
    def exclusion_meta(self):
        """Get the filtered exclusions meta data.

        Returns
        -------
        excl_points : pd.DataFrame
            Exclusions meta data reduced to just the exclusion points
            associated with the current supply curve point. Also gets filtered
            to just exclusion points with corresponding generation points.
        """
        return self._excl_meta

    def get_base_excl_meta(self):
        """Get the base exclusions meta data for all excl point in SC point.

        Returns
        -------
        base_excl_meta : pd.DataFrame
            Exclusions meta data reduced to just the exclusion points
            associated with the current supply curve point.
        """
        base_excl_meta = self.exclusions['meta',
                                         self._excl_row_slice,
                                         self._excl_col_slice]
        return base_excl_meta

    @property
    def distance_upper_bound(self):
        """Get the upper bound on NN distance between excl and gen points.

        Returns
        -------
        distance_upper_bound : float
            Estimate of the upper bound distance based on the distance between
            generation points. Calculated as half of the diagonal between
            closest generation points, with an extra 5% margin.
        """
        if self._distance_upper_bound is None:
            lats = self.gen.meta.loc[self.gen_mask, 'latitude'].values
            dists = np.abs(lats - (lats[0] * np.ones_like(lats)))
            dists = dists[(dists != 0)]
            self._distance_upper_bound = 1.05 * (2 ** 0.5) * (dists.min() / 2)
        return self._distance_upper_bound

    @property
    def lat_lon_lims(self):
        """Get the supply curve point lat/lon limits.

        Format is ((lat_min, lat_max), (lon_min, lon_max))

        Returns
        -------
        lat_lon_limts : tuple
            Minimum/maximum latitude/longitude value in the exclusion points
            for this sc point.
        """

        if self._lat_lon_lims is None:
            meta = self.get_base_excl_meta()
            lat_min = meta['latitude'].min()
            lat_max = meta['latitude'].max()
            lon_min = meta['longitude'].min()
            lon_max = meta['longitude'].max()
            self._lat_lon_lims = ((lat_min, lat_max), (lon_min, lon_max))

        return self._lat_lon_lims

    @property
    def centroid(self):
        """Get the supply curve point centroid coordinate.

        Returns
        -------
        centroid : tuple
            SC point centroid (lat, lon).
        """

        centroid = (np.sum(self.lat_lon_lims[0]) / 2,
                    np.sum(self.lat_lon_lims[1]) / 2)
        return centroid

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
    def gen_mask(self):
        """Get the generation mask based on coordinates close to excl points.

        Returns
        -------
        _gen_mask : pd.Series
            Boolean mask for the generation sites close to the exclusion
            points.
        """
        if self._gen_mask is None:
            # margin is the extra distance (in decimal lat/lon) to allow
            margin = 0.5
            lat_min = np.min(self.lat_lon_lims[0])
            lat_max = np.max(self.lat_lon_lims[0])
            lon_min = np.min(self.lat_lon_lims[1])
            lon_max = np.max(self.lat_lon_lims[1])
            self._gen_mask = ((self.gen.meta['latitude'] > lat_min - margin) &
                              (self.gen.meta['latitude'] < lat_max + margin) &
                              (self.gen.meta['longitude'] > lon_min - margin) &
                              (self.gen.meta['longitude'] < lon_max + margin))

            if self._gen_mask.sum() == 0:
                msg = ('Supply curve point with row/col slices {}/{} at '
                       'centroid lat/lon {} has no viable generation points '
                       'based on exclusion and gen files: "{}", "{}"'
                       .format(self._excl_row_slice, self._excl_col_slice,
                               self.centroid, self._fpath_excl,
                               self._fpath_gen))
                raise EmptySupplyCurvePointError(msg)
        return self._gen_mask

    @property
    def gen_tree(self):
        """Get the generation meta ckdtree.

        Returns
        -------
        _gen_tree : ckdtree
            Spatial ckdtree built on the generation meta data.
        """
        if self._gen_tree is None:
            self._gen_tree = cKDTree(
                self.gen.meta.loc[self.gen_mask, ['latitude', 'longitude']])
        return self._gen_tree


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

        self._res = resolution
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
