# -*- coding: utf-8 -*-
"""Output request config to handle user output requests.

This module will allow for aliases and fix some typos.

Created on Mon Jul  8 09:37:23 2019

@author: gbuster
"""
from warnings import warn
from reV.utilities.exceptions import ConfigWarning


class OutputRequest(list):
    """Base output request list framework with request key correction logic."""

    # map of commonly expected typos.
    # keys are typos, values are correct var names
    # all available output variables should be in the values
    CORRECTIONS = {}

    def __init__(self, inp):
        """
        Parameters
        ----------
        inp : list | tuple | str
            List of requested reV output variables.
        """

        if isinstance(inp, str):
            inp = [inp]

        for request in inp:
            if request in self.CORRECTIONS.values():
                self.append(request)
            elif request in self.CORRECTIONS.keys():
                self.append(self.CORRECTIONS[request])
                warn('Correcting output request "{}" to "{}".'
                     .format(request, self.CORRECTIONS[request]),
                     ConfigWarning)
            else:
                self.append(request)
                warn('Did not recognize requested output variable "{}". '
                     'Passing forward, but this may cause a downstream '
                     'error. Available known output variables are: {}'
                     .format(request, list(set(self.CORRECTIONS.values()))),
                     ConfigWarning)


class SAMOutputRequest(OutputRequest):
    """SAM output request framework."""

    # map of commonly expected typos.
    # keys are typos, values are correct SAM var names
    # all available SAM output variables should be in the values
    CORRECTIONS = {'cf_means': 'cf_mean',
                   'cf': 'cf_mean',
                   'capacity_factor': 'cf_mean',
                   'capacityfactor': 'cf_mean',
                   'cf_profiles': 'cf_profile',
                   'profiles': 'cf_profile',
                   'profile': 'cf_profile',
                   'dni_means': 'dni_mean',
                   'ghi_means': 'ghi_mean',
                   'ws_means': 'ws_mean',
                   'generation': 'annual_energy',
                   'yield': 'energy_yield',
                   'generation_profile': 'gen_profile',
                   'generation_profiles': 'gen_profile',
                   'plane_of_array': 'poa',
                   'plane_of_array_irradiance': 'poa',
                   'gen_profiles': 'gen_profile',
                   'lcoe': 'lcoe_fcr',
                   'lcoe_nominal': 'lcoe_nom',
                   'real_lcoe': 'lcoe_real',
                   'net_present_value': 'project_return_aftertax_npv',
                   'npv': 'project_return_aftertax_npv',
                   'ppa': 'ppa_price',
                   'single_owner': 'ppa_price',
                   'singleowner': 'ppa_price',
                   'actual_irr': 'flip_actual_irr',
                   'irr': 'flip_actual_irr',
                   'cf_total_revenue': 'gross_revenue',
                   'total_cost': 'total_installed_cost',
                   'turbine': 'turbine_cost',
                   'sales_tax': 'sales_tax_cost',
                   'bos': 'bos_cost',
                   }
