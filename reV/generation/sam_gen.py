"""
SAM derived generation
"""


class SAM_Gen(object):
    """
    Base class for SAM derived generation
    """
    pass


class Solar(SAM_Gen):
    """
    Base Class for Solar generation from SAM
    """
    pass


class PV(Solar):
    """
    PV generation
    """
    pass


class CHP(Solar):
    """
    CHP generation
    """
    pass


class Wind(SAM_Gen):
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
