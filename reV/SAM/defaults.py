# -*- coding: utf-8 -*-
"""
PySAM default execution objects.
"""

import os
from reV import TESTDATADIR
import PySAM.Pvwattsv5 as pysam_pv
import PySAM.Windpower as pysam_wind
import PySAM.TcsmoltenSalt as pysam_csp
import PySAM.Lcoefcr as pysam_lcoe
import PySAM.Singleowner as pysam_so


def pvwattsv5():
    """Make a default pysam object for pvwatts"""
    res_file = os.path.join(
        TESTDATADIR, 'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
    x = pysam_pv.default('PVWattsNone')
    x.LocationAndResource.solar_resource_file = res_file
    x.execute()
    return x


def windpower():
    """Make a default pysam object for windpower"""
    res_file = os.path.join(
        TESTDATADIR, 'SAM/WY Southern-Flat Lands.csv')
    x = pysam_wind.default('WindPowerNone')
    x.WindResourceFile.wind_resource_filename = res_file
    x.execute()
    return x


def csp():
    """Make a default pysam object for csp"""
    res_file = os.path.join(
        TESTDATADIR, 'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
    x = pysam_csp.default('MSPTSingleOwner')
    x.LocationAndResource.solar_resource_file = res_file
    x.execute()
    return x


def lcoe():
    """Make a default pysam object for lcoe"""
    res_file = os.path.join(
        TESTDATADIR, 'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
    x = pysam_pv.default('PVWattsLCOECalculator')
    x.LocationAndResource.solar_resource_file = res_file
    x.execute()

    y = pysam_lcoe.default('PVWattsLCOECalculator')
    y.SimpleLCOE.annual_energy = x.Outputs.annual_energy
    y.execute()
    return y


def single_owner():
    """Make a default pysam object for single owner"""
    res_file = os.path.join(
        TESTDATADIR, 'SAM/USA AZ Phoenix Sky Harbor Intl Ap (TMY3).csv')
    x = pysam_pv.default('PVWattsSingleOwner')
    x.LocationAndResource.solar_resource_file = res_file
    x.execute()

    y = pysam_so.default('PVWattsSingleOwner')
    y.SystemOutput.gen = x.Outputs.ac
    y.execute()
    return y


if __name__ == '__main__':
    a = pvwattsv5()
#    b = windpower()
#    c = csp()
#    d = lcoe()
#    e = single_owner()
