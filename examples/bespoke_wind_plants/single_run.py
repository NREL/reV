# -*- coding: utf-8 -*-
"""
An example single run to get bespoke wind plant layout
"""
import json
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeSinglePlant
from reV.bespoke.plotting_functions import (
    plot_poly,
    plot_turbines,
    plot_windrose,
)
from reV.supply_curve.tech_mapping import TechMapping
from reV.utilities import MetaKeyName

SAM = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')
EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
TM_DSET = 'techmap_wtk_ri_100'
AGG_DSET = (MetaKeyName.CF_MEAN, MetaKeyName.CF_PROFILE)

# note that this differs from the
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': False},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': False},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': False}}

with open(SAM) as f:
    SAM_SYS_INPUTS = json.load(f)

SAM_SYS_INPUTS['wind_farm_wake_model'] = 2
SAM_SYS_INPUTS['wind_farm_losses_percent'] = 0
del SAM_SYS_INPUTS['wind_resource_filename']
TURB_RATING = np.max(SAM_SYS_INPUTS['wind_turbine_powercurve_powerout'])
SAM_CONFIGS = {'default': SAM_SYS_INPUTS}


# def cost_function(x):
#     """dummy cost function"""
#     R = 0.1
#     return 200 * x * np.exp(-x / 1E5 * R + (1 - R))


# def objective_function(aep, cost):
#     """dummy objective function"""
#     return cost / aep


if __name__ == "__main__":

    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"

    output_request = ('system_capacity', MetaKeyName.CF_MEAN,
                      MetaKeyName.CF_PROFILE)
    gid = 33
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        bsp = BespokeSinglePlant(gid, excl_fp, res_fp, TM_DSET,
                                 SAM_SYS_INPUTS,
                                 objective_function, cost_function,
                                 ga_kwargs={'max_time': 20},
                                 excl_dict=EXCL_DICT,
                                 output_request=output_request,
                                 )
        results = bsp.run_plant_optimization()

    # print(results)
    # print(type(results))
    # print(results.keys())
    print("nturbs: ", results["n_turbines"])
    print("system_capacity: ", results["system_capacity"])
    # print("non_excluded_area: ", results["non_excluded_area"])
    # print("non_excluded_capacity_density: ",
    #       results["non_excluded_capacity_density"])
    print("bespoke_aep: ", results["bespoke_aep"])
    print("bespoke_objective: ", results["bespoke_objective"])
    print("bespoke_annual_cost: ", results["bespoke_annual_cost"])

    rotor_diameter = bsp.sam_sys_inputs["wind_turbine_rotor_diameter"]

    ax = plot_windrose(np.arange(bsp._wd_bins[0], bsp._wd_bins[1],
                       bsp._wd_bins[2]),
                       np.arange(bsp._ws_bins[0], bsp._ws_bins[1],
                       bsp._ws_bins[2]),
                       bsp._wind_dist)
    ax.set_title("wind rose")

    ax = plot_poly(results["full_polygons"])
    ax = plot_turbines(bsp.plant_optimizer.turbine_x,
                       bsp.plant_optimizer.turbine_y,
                       rotor_diameter / 2, ax=ax)
    ax.axis("equal")
    ax.set_title("full polys, turbines")

    ax = plot_poly(results["packing_polygons"])
    ax = plot_turbines(bsp.plant_optimizer.x_locations,
                       bsp.plant_optimizer.y_locations,
                       rotor_diameter / 2, ax=ax)
    ax.axis("equal")
    ax.set_title("packed polys, packed points")

    plt.show()
