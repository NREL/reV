# -*- coding: utf-8 -*-
"""
Classes to handle reV h5 output files.
"""
import json
import logging
import sys

import NRWAL
import PySAM
import rex
from rex.outputs import Outputs as rexOutputs

from reV.version import __version__

logger = logging.getLogger(__name__)


class Outputs(rexOutputs):
    """
    Base class to handle reV output data in .h5 format

    Examples
    --------
    The reV Outputs handler can be used to initialize h5 files in the standard
    reV/rex resource data format.

    >>> from reV import Outputs
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> meta = pd.DataFrame({SupplyCurveField.LATITUDE: np.ones(100),
    >>>                      SupplyCurveField.LONGITUDE: np.ones(100)})
    >>>
    >>> time_index = pd.date_range('20210101', '20220101', freq='1h',
    >>>                            closed='right')
    >>>
    >>> with Outputs('test.h5', 'w') as f:
    >>>     f.meta = meta
    >>>     f.time_index = time_index

    You can also use the Outputs handler to read output h5 files from disk.
    The Outputs handler will automatically parse the meta data and time index
    into the expected pandas objects (DataFrame and DatetimeIndex,
    respectively).

    >>> with Outputs('test.h5') as f:
    >>>     print(f.meta.head())
    >>>
         latitude  longitude
    gid
    0         1.0        1.0
    1         1.0        1.0
    2         1.0        1.0
    3         1.0        1.0
    4         1.0        1.0

    >>> with Outputs('test.h5') as f:
    >>>     print(f.time_index)
    DatetimeIndex(['2021-01-01 01:00:00+00:00', '2021-01-01 02:00:00+00:00',
                   '2021-01-01 03:00:00+00:00', '2021-01-01 04:00:00+00:00',
                   '2021-01-01 05:00:00+00:00', '2021-01-01 06:00:00+00:00',
                   '2021-01-01 07:00:00+00:00', '2021-01-01 08:00:00+00:00',
                   '2021-01-01 09:00:00+00:00', '2021-01-01 10:00:00+00:00',
                   ...
                   '2021-12-31 15:00:00+00:00', '2021-12-31 16:00:00+00:00',
                   '2021-12-31 17:00:00+00:00', '2021-12-31 18:00:00+00:00',
                   '2021-12-31 19:00:00+00:00', '2021-12-31 20:00:00+00:00',
                   '2021-12-31 21:00:00+00:00', '2021-12-31 22:00:00+00:00',
                   '2021-12-31 23:00:00+00:00', '2022-01-01 00:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', length=8760, freq=None)

    There are a few ways to use the Outputs handler to write data to a file.
    Here is one example using the pre-initialized file we created earlier.
    Note that the Outputs handler will automatically scale float data using
    the "scale_factor" attribute. The Outputs handler will unscale the data
    while being read unless the unscale kwarg is explicityly set to False.
    This behavior is intended to reduce disk storage requirements for big
    data and can be disabled by setting dtype=np.float32 or dtype=np.float64
    when writing data.

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='dset1',
    >>>                     dset_data=np.ones((8760, 100)) * 42.42,
    >>>                     attrs={'scale_factor': 100}, dtype=np.int32)


    >>> with Outputs('test.h5') as f:
    >>>     print(f['dset1'])
    >>>     print(f['dset1'].dtype)
    [[42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     ...
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]
     [42.42 42.42 42.42 ... 42.42 42.42 42.42]]
    float32

    >>> with Outputs('test.h5', unscale=False) as f:
    >>>     print(f['dset1'])
    >>>     print(f['dset1'].dtype)
    [[4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     ...
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]
     [4242 4242 4242 ... 4242 4242 4242]]
    int32

    Note that the reV Outputs handler is specifically designed to read and
    write spatiotemporal data. It is therefore important to intialize the meta
    data and time index objects even if your data is only spatial or only
    temporal. Furthermore, the Outputs handler will always assume that 1D
    datasets represent scalar data (non-timeseries) that corresponds to the
    meta data shape, and that 2D datasets represent spatiotemporal data whose
    shape corresponds to (len(time_index), len(meta)). You can see these
    constraints here:

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='bad_shape',
                            dset_data=np.ones((1, 100)) * 42.42,
                            attrs={'scale_factor': 100}, dtype=np.int32)
    HandlerValueError: 2D data with shape (1, 100) is not of the proper
    spatiotemporal shape: (8760, 100)

    >>> Outputs.add_dataset(h5_file='test.h5', dset_name='bad_shape',
                            dset_data=np.ones((8760,)) * 42.42,
                            attrs={'scale_factor': 100}, dtype=np.int32)
    HandlerValueError: 1D data with shape (8760,) is not of the proper
    spatial shape: (100,)
    """

    @property
    def full_version_record(self):
        """Get record of versions for dependencies

        Returns
        -------
        dict
            Dictionary of package versions for dependencies
        """
        rev_versions = {'reV': __version__,
                        'rex': rex.__version__,
                        'pysam': PySAM.__version__,
                        'python': sys.version,
                        'nrwal': NRWAL.__version__,
                        }
        versions = super().full_version_record
        versions.update(rev_versions)
        return versions

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
        self.h5.attrs['full_version_record'] = json.dumps(
            self.full_version_record)
        self.h5.attrs['package'] = 'reV'
