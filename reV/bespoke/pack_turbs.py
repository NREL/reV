# -*- coding: utf-8 -*-
"""
turbine packing module.
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from reV.bespoke.plotting_functions import get_xy


class PackTurbines():
    """Framework to maximize plant capacity in a provided wind plant area.
    """

    def __init__(self, min_spacing, safe_polygons, weight_x=0.0):
        """
        Parameters
        ----------
        min_spacing : float
            The minimum allowed spacing between wind turbines.
        safe_polygons : Polygon | MultiPolygon
            The "safe" area(s) where turbines can be placed without
            violating boundary, setback, exclusion, or other constraints.
        weight_x : float, optional
            A weight used to alter the angle at which turbine rows are created.
        turbine_x : array
            Packed turbine x coordinates.
        turbine_y : array
            Packed turbine y coordinates.
        """

        self.min_spacing = min_spacing
        self.safe_polygons = safe_polygons
        self.weight_x = weight_x

        # turbine locations
        self.turbine_x = np.array([])
        self.turbine_y = np.array([])

    def pack_turbines_poly(self):
        """Fast packing algorithm that maximizes plant capacity in a
        provided wind plant area.
        """

        can_add_more = True
        leftover = MultiPolygon(self.safe_polygons)
        while can_add_more:
            nareas = len(leftover)
            if nareas > 0:
                areas = np.zeros(len(leftover))
                for i in range(nareas):
                    areas[i] = leftover[i].area
                smallest_area = leftover[np.argmin(areas)]
                exterior_coords = smallest_area.exterior.coords[:]
                x, y = get_xy(exterior_coords)
                metric = self.weight_x * x + y
                self.turbine_x = np.append(self.turbine_x,
                                           x[np.argmin(metric)])
                self.turbine_y = np.append(self.turbine_y,
                                           y[np.argmin(metric)])
                new_turbine = Point(x[np.argmin(metric)], y[np.argmin(metric)]
                                    ).buffer(self.min_spacing)
            else:
                can_add_more = False

            leftover = leftover.difference(new_turbine)
            if isinstance(leftover, Polygon):
                leftover = MultiPolygon([leftover])

    def clear(self):
        """Reset the packing algorithm by clearing the x and y turbine arrays
        """
        self.turbine_x = np.array([])
        self.turbine_y = np.array([])
