# -*- coding: utf-8 -*-
"""
Exclusion layers handler
"""
import h5py
import logging

logger = logging.getLogger(__name__)


class ExclusionLayers:
    """
    Handler of .h5 file and techmap for Exclusion Layers
    """
    def __init__(self, h5_file, hsds=False):
        """
        Parameters
        ----------
        h5_file : str
            .h5 file containing exclusion layers and techmap
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self.h5_file = h5_file
        if hsds:
            import h5pyd
            self._h5 = h5pyd.File(self.h5_file, 'r')
        else:
            self._h5 = h5py.File(self.h5_file, 'r')
        self._meta = None
