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
        Exterior coordinates from a shapely Polygon

    Outputs
    ----------
    x, y : array
        Boundary polygon x and y coordinates
    """
    x = np.zeros(len(A))
    y = np.zeros(len(A))
    for i, _ in enumerate(A):
        x[i] = A[i][0]
        y[i] = A[i][1]
    return x, y


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
