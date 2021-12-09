# -*- coding: utf-8 -*-
"""
functions to plot turbine layouts and boundary polygons
"""
import numpy as np
import matplotlib.pyplot as plt


def get_xy(A):
    """separate polygon exterior coordinates to x and y
    Parameters
    ----------
    A : Polygon.exteroir.coords
    exterior coordinates from a shapely Polygon

    Outputs
    ----------
    x : array
    boundary polygon x coordinates
    y : array
    boundary polygon y coordinates
    """
    x = np.zeros(len(A))
    y = np.zeros(len(A))
    for i, _ in enumerate(A):
        x[i] = A[i][0]
        y[i] = A[i][1]
    return x, y


def plot_poly(geom, ax=False, color="black", linestyle="--", linewidth=0.5):
    """plot the wind plant boundaries
    Parameters
    ----------
    geom : Polygon | MultiPolygon
    the shapely.Polygon or shapely.MultiPolygon that define the wind plant
    boundary(ies).
    ax : matplotlib axis (optional)
    the axis you want to plot on
    color : string (optional)
    the color for the wind plant boundaries
    linestyle : string (optional)
    style to plot the boundary lines
    linewidth : float (optional)
    the width of the boundary lines
    """
    if ax is False:
        ax = plt.gca()
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        x, y = get_xy(exterior_coords)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)

        for interior in geom.interiors:
            interior_coords = interior.coords[:]
            x, y = get_xy(interior_coords)
            ax.plot(x, y, "--b", linewidth=0.5)

    elif geom.type == 'MultiPolygon':

        for part in geom:
            exterior_coords = part.exterior.coords[:]
            x, y = get_xy(exterior_coords)
            ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=linewidth)

            for interior in part.interiors:
                interior_coords = interior.coords[:]
                x, y = get_xy(interior_coords)
                ax.plot(x, y, "--b", linewidth=0.5)


def plot_turbines(x, y, r, color="C0", nums=False):
    """plot wind turbine locations
    Parameters
    ----------
    x : array
    wind turbine x locations
    y : array
    wind turbine y locations
    r : float
    wind turbine radius
    color : string (optional)
    the color for the wind plant boundaries
    nums : Bool (optional)
    show the turbine numbers next to each turbine
    """
    n = len(x)
    for i in range(n):
        t = plt.Circle((x[i], y[i]), r, color=color)
        plt.gca().add_patch(t)
        if nums is True:
            plt.text(x[i], y[i], "%s" % (i + 1))
