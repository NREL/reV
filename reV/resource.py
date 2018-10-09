"""
Classes to handle resource data
"""
#  import h5py


class Resource(object):
    """
    Base class to handle resource .h5 files
    """
    pass


class NSRDB(Resource):
    """
    Class to handle NSRDB .h5 files
    """
    pass


class WTK(Resource):
    """
    Class to handle WTK .h5 files
    """
    pass


class ECMWF(Resource):
    """
    Class to handle ECMWF weather forecast .h5 files
    """
    pass


class MERRA2(Resource):
    """
    Class to handle MERRA2 .h5 files
    """
    pass


class ERA5(Resource):
    """
    Class to handle ERA5 .h5 files
    """
    pass
