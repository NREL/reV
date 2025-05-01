# -*- coding: utf-8 -*-
"""
An example single run to get bespoke wind plant layout
"""
import json
import os
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from reV import TESTDATADIR
from reV.bespoke.pack_turbs import get_xy
from reV.bespoke.bespoke import BespokeSinglePlant
from reV.supply_curve.tech_mapping import TechMapping

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


def plot_poly(geom, ax=None, color="black", linestyle="--", linewidth=0.5):
    """plot the wind plant boundaries

    Parameters
    ----------
    geom : Polygon | MultiPolygon
        The shapely.Polygon or shapely.MultiPolygon that define the wind
        plant boundary(ies).
    ax : :py:class:`matplotlib.pyplot.axes`, optional
        The figure axes on which the wind rose is plotted.
        Defaults to :obj:`None`.
    color : string, optional
        The color for the wind plant boundaries
    linestyle : string, optional
        Style to plot the boundary lines
    linewidth : float, optional
        The width of the boundary lines
    """
    if ax is None:
        _, ax = plt.subplots()

    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        x, y = get_xy(exterior_coords)
        ax.fill(x, y, color="C0", alpha=0.25)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)

        for interior in geom.interiors:
            interior_coords = interior.coords[:]
            x, y = get_xy(interior_coords)
            ax.fill(x, y, color="white", alpha=1.0)
            ax.plot(x, y, "--k", linewidth=0.5)

    elif geom.type == 'MultiPolygon':

        for part in geom:
            exterior_coords = part.exterior.coords[:]
            x, y = get_xy(exterior_coords)
            ax.fill(x, y, color="C0", alpha=0.25)
            ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=linewidth)

            for interior in part.interiors:
                interior_coords = interior.coords[:]
                x, y = get_xy(interior_coords)
                ax.fill(x, y, color="white", alpha=1.0)
                ax.plot(x, y, "--k", linewidth=0.5)
    return ax


def plot_turbines(x, y, r, ax=None, color="C0", nums=False):
    """plot wind turbine locations

    Parameters
    ----------
    x, y : array
        Wind turbine x and y locations
    r : float
        Wind turbine radius
    ax :py:class:`matplotlib.pyplot.axes`, optional
        The figure axes on which the wind rose is plotted.
        Defaults to :obj:`None`.
    color : string, optional
        The color for the wind plant boundaries
    nums : bool, optional
        Option to show the turbine numbers next to each turbine
    """
    # Set up figure
    if ax is None:
        _, ax = plt.subplots()

    n = len(x)
    for i in range(n):
        t = plt.Circle((x[i], y[i]), r, color=color)
        ax.add_patch(t)
        if nums is True:
            ax.text(x[i], y[i], "%s" % (i + 1))

    return ax


def plot_windrose(wind_directions, wind_speeds, wind_frequencies, ax=None,
                  colors=None):
    """plot windrose

    Parameters
    ----------
    wind_directions : 1D array
        Wind direction samples
    wind_speeds : 1D array
        Wind speed samples
    wind_frequencies : 2D array
        Frequency of wind direction and speed samples
    ax :py:class:`matplotlib.pyplot.axes`, optional
        The figure axes on which the wind rose is plotted.
        Defaults to :obj:`None`.
    color : array, optional
        The color for the different wind speed bins
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw=dict(polar=True))

    ndirs = len(wind_directions)
    nspeeds = len(wind_speeds)

    if colors is None:
        colors = []
        for i in range(nspeeds):
            colors = np.append(colors, "C%s" % i)

    for i in range(ndirs):
        wind_directions[i] = np.deg2rad(90.0 - wind_directions[i])

    width = 0.8 * 2 * np.pi / len(wind_directions)

    for i in range(ndirs):
        bottom = 0.0
        for j in range(nspeeds):
            if i == 0:
                if j < nspeeds - 1:
                    ax.bar(wind_directions[i], wind_frequencies[j, i],
                           bottom=bottom, width=width, edgecolor="black",
                           color=[colors[j]],
                           label="%s-%s m/s" % (int(wind_speeds[j]),
                                                int(wind_speeds[j + 1]))
                           )
                else:
                    ax.bar(wind_directions[i], wind_frequencies[j, i],
                           bottom=bottom, width=width, edgecolor="black",
                           color=[colors[j]],
                           label="%s+ m/s" % int(wind_speeds[j])
                           )
            else:
                ax.bar(wind_directions[i], wind_frequencies[j, i],
                       bottom=bottom, width=width, edgecolor="black",
                       color=[colors[j]])
            bottom = bottom + wind_frequencies[j, i]

    ax.legend(bbox_to_anchor=(1.3, 1), fontsize=10)
    pi = np.pi
    ax.set_xticks((0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                   3 * pi / 2, 7 * pi / 4))
    ax.set_xticklabels(("E", "NE", "N", "NW", "W", "SW", "S", "SE"),
                       fontsize=10)
    plt.yticks(fontsize=10)

    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)

    return ax


if __name__ == "__main__":

    cost_function = """200 * system_capacity * np.exp(-system_capacity /
        1E5 * 0.1 + (1 - 0.1))"""
    objective_function = "cost / aep"

    output_request = ('system_capacity', 'cf_mean',
                      'cf_profile')
    gid = 33
    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(
            excl_fp, RES.format(2012), tm_dset=TM_DSET, max_workers=1
        )
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
