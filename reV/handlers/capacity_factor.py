"""
Classes to handle capacity factor profiles and annual averages
"""
import h5py
from reV.handler.resource import Resource, parse_keys


class CapacityFactor(Resource):
    """
    Base class to handle capacity factor data in .h5 format
    """
    def __init__(self, h5_file, unscale=True, mode='r'):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        """
        self._h5_file = h5_file
        self._mode = mode
        self._unscale = unscale

    def __enter__(self):
        self.open()
        return self

    def __len__(self):
        if self.hasattr('_h5'):
            len = super(CapacityFactor).len(self)
        else:
            len = None

        return len

    def __setitem__(self, keys, arr):
        mode = ['a', 'w', 'w-', 'x']
        msg = 'mode must be writable: {}'.format(mode)
        assert self._mode in mode, msg

        ds, ds_slice = parse_keys(keys)

        if ds == 'meta':
            self.meta = arr
        elif ds == 'time_index':
            self.time_index = arr
        else:
            self._set_ds_array(ds, arr, *ds_slice)

    def _set_ds_array(self, ds, arr, *ds_slice):
        """
        Write ds to disc
        """
        pass

    def open(self):
        """
        Initialize h5py File instance
        """
        self._h5 = h5py.File(self._h5_path, mode=self._mode)
        self._dsets = list(self._h5)

    def close(self):
        """
        Close h5 instance
        """
        if self.hasattr('_h5'):
            self._h5.close()
