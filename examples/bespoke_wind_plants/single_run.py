# -*- coding: utf-8 -*-
"""
An example single run to get bespoke wind plant layout
"""
import numpy as np
import matplotlib.pyplot as plt
from reV.bespoke.bespoke import BespokeWindFarms
from reV.bespoke.plotting_functions import plot_poly, plot_turbines,\
    plot_windrose
from reV import TESTDATADIR
from reV.supply_curve.tech_mapping import TechMapping

import json
import os
import shutil
import tempfile

SAM = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')
EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
TM_DSET = 'techmap_wtk_ri_100'

# note that this differs from the
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': False},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': False},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': False}}


if __name__ == '__main__':

    def cost_function(x):
        """dummy cost function"""
        R = 0.1
        return 200 * x * np.exp(-x / 1E5 * R + (1 - R))

    def objective_function(aep, cost):
        """dummy objective function"""
        return cost / aep

    ga_time = 20.0

    with open(SAM, 'r') as f:
        sam_sys_inputs = json.load(f)

    rotor_diameter = sam_sys_inputs["wind_turbine_rotor_diameter"]
    min_spacing = 5 * rotor_diameter

    gid = 34  # 39% included
    # gid = 34
    ws_dset = 'windspeed_88m'
    wd_dset = 'winddirection_88m'

    with open(SAM, 'r') as f:
        sam_sys_inputs = json.load(f)

    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET,
                        max_workers=1)
        out = BespokeWindFarms.run_serial(excl_fp, res_fp, TM_DSET, ws_dset,
                                          wd_dset, sam_sys_inputs,
                                          objective_function, cost_function,
                                          min_spacing, ga_time,
                                          excl_dict=EXCL_DICT, gids=gid)

    results = out[gid]
    print("nturbs: ", results["nturbs"])
    print("plant_capacity: ", results["plant_capacity"])
    print("non_excluded_area: ", results["non_excluded_area"])
    print("non_excluded_capacity_density: ",
          results["non_excluded_capacity_density"])
    print("aep: ", results["aep"])
    print("objective: ", results["objective"])
    print("annual_cost: ", results["annual_cost"])

    plt.figure(1)
    ax = plot_windrose(results["wd_sample_points"],
                       results["ws_sample_points"],
                       results["wind_dist"])
    plt.title("wind rose")

    plt.figure(2)
    ax = plot_poly(results["full_polygons"])
    ax = plot_turbines(results["packed_x"],
                       results["packed_y"], rotor_diameter / 2,
                       ax=ax)
    plt.axis("equal")
    plt.title("full polys, packed points")

    plt.figure(3)
    ax = plot_poly(results["full_polygons"])
    ax = plot_turbines(results["turbine_x_coords"],
                       results["turbine_y_coords"], rotor_diameter / 2,
                       ax=ax)
    plt.axis("equal")
    plt.title("full polys, turbines")

    plt.figure(4)
    ax = plot_poly(results["packing_polygons"])
    ax = plot_turbines(results["packed_x"],
                       results["packed_y"], rotor_diameter / 2,
                       ax=ax)
    plt.axis("equal")
    plt.title("packed polys, packed points")

    plt.figure(5)
    ax = plot_poly(results["packing_polygons"])
    ax = plot_turbines(results["turbine_x_coords"],
                       results["turbine_y_coords"], rotor_diameter / 2,
                       ax=ax)
    plt.axis("equal")
    plt.title("packed polys, turbines")

    plt.show()
