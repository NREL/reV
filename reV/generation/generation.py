"""
Generation
"""


class Gen(object):
    """
    Base class for generation
    """
    pass


class Solar(Gen):
    """
    Base Class for Solar generation
    """
    pass


class PV(Solar):
    """
    PV generation
    """
    pass


class CSP(Solar):
    """
    CSP generation
    """
    pass


class Wind(Gen):
    """
    Base class for Wind generation from SAM
    """
    pass


class LandBasedWind(Wind):
    """
    Onshore wind generation
    """
    pass


class OffshoreWind(Wind):
    """
    Offshore wind generation
    """
    pass
