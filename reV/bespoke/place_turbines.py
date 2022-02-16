# -*- coding: utf-8 -*-
"""
place turbines for bespoke wind plants
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import shapely.affinity
import rasterio.features

from reV.bespoke.pack_turbs import PackTurbines
from reV.bespoke.gradient_free import GeneticAlgorithm


class PlaceTurbines():
    """Framework for optimizing turbine locations for site specific
    exclusions, wind resources, and objective
    """
    def __init__(self, wind_plant, objective_function, cost_function,
                 include_mask, pixel_side_length, min_spacing, ga_time):
        """
        Parameters
        ----------
        wind_plant : WindPowerPD
            wind plant object to analyze wind plant performance. This object
            should have everything in the plant defined, such that only the
            turbine coordinates and plant capacity need to be defined during
            the optimization.
        objective_function: function(aep, cost)
            the objective function for wind plant layout optimization. Should
            be a function of the plant aep and annual cost. As per optimization
            convention, this function will be minimized.
        cost_function : function(capacity)
            the cost function for wind plant layout optimization. Should
            be a function of the plant capacity, and returns the annual
            plant costs.
        exclusions : ExclusionMaskFromDict
            The exclusions that define where turbines can be placed. Contains
            exclusions.latitude, exclusions.longitude, and exclusions.mask
        min_spacing : float
            The minimum spacing between turbines (in meters).
        ga_time : float
            The time to run the genetic algorithm (in seconds).
        """

        # need to be assigned
        self.wind_plant = wind_plant
        self.cost_function = cost_function
        self.objective_function = objective_function
        self.include_mask = include_mask
        self.pixel_side_length = pixel_side_length
        self.min_spacing = min_spacing
        self.ga_time = ga_time
        # internal variables
        self.nrows, self.ncols = np.shape(include_mask)
        self.x_locations = np.array([])
        self.y_locations = np.array([])
        self.turbine_capacity = \
            np.max(self.wind_plant.
                   sam_sys_inputs["wind_turbine_powercurve_powerout"])
        self.full_polygons = None
        self.packing_polygons = None

        # outputs
        self.turbine_x = np.array([])
        self.turbine_y = np.array([])
        self.nturbs = 0
        self.capacity = 0.0
        self.area = 0.0
        self.capacity_density = 0.0
        self.aep = 0.0
        self.objective = 0.0

    def define_exclusions(self):
        """From the exclusions data, create a shapely MultiPolygon as
        self.safe_polygons that defines where turbines can be placed.
        """

        shapes = rasterio.features.shapes(np.floor(self.include_mask))
        polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes
                    if shape[1] == 1]
        for i, _ in enumerate(polygons):
            polygons[i] = shapely.affinity.scale(polygons[i],
                                                 xfact=self.pixel_side_length,
                                                 yfact=-self.pixel_side_length,
                                                 origin=(0, 0))

        safe_polygons = MultiPolygon(polygons)
        minx, miny, maxx, maxy = safe_polygons.bounds
        safe_polygons = shapely.affinity.translate(safe_polygons, xoff=-minx,
                                                   yoff=-miny)
        self.full_polygons = safe_polygons.buffer(0)

        # add extra setback to cell boundary
        minx, miny, maxx, maxy = self.full_polygons.bounds
        minx += self.min_spacing / 2.0
        miny += self.min_spacing / 2.0
        maxx -= self.min_spacing / 2.0
        maxy -= self.min_spacing / 2.0

        boundary_poly = \
            Polygon(((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)))
        self.packing_polygons = boundary_poly.intersection(self.full_polygons)

    def initialize_packing(self):
        """run the turbine packing algorithm (maximizing plant capacity) to
        define potential turbine locations that will be used as design
        variables in the gentic algorithm.
        """

        packing = PackTurbines(self.min_spacing, self.packing_polygons)
        nturbs = 1E6
        mult = 1.0
        while nturbs > 300:
            packing.clear()
            packing.min_spacing = self.min_spacing * mult
            packing.pack_turbines_poly()
            nturbs = len(packing.turbine_x)
            mult *= 1.1
        self.x_locations = packing.turbine_x
        self.y_locations = packing.turbine_y

    def optimization_objective(self, x):
        """The optimization objective used in the bespoke optimization
        """
        x = [bool(y) for y in x]
        n_turbines = np.sum(x)
        self.wind_plant["wind_farm_xCoordinates"] = self.x_locations[x]
        self.wind_plant["wind_farm_yCoordinates"] = self.y_locations[x]

        system_capacity = n_turbines * self.turbine_capacity
        self.wind_plant["system_capacity"] = system_capacity

        self.wind_plant.assign_inputs()
        self.wind_plant.execute()
        aep = self.wind_plant.annual_energy()
        cost = self.cost_function(system_capacity)

        return self.objective_function(aep, cost)

    def optimize(self):
        """use a genetic algorithm to optimize wind plant layout for the user
        defined objective function.
        """
        nlocs = len(self.x_locations)
        bits = np.ones(nlocs, dtype=int)
        bounds = np.zeros((nlocs, 2), dtype=int)
        bounds[:, 1] = 2
        variable_type = np.array([])
        for _ in range(nlocs):
            variable_type = np.append(variable_type, "int")
        ga = GeneticAlgorithm(bits, bounds, variable_type,
                              self.optimization_objective)

        ga.max_generation = 10000
        ga.population_size = 25
        ga.crossover_rate = 0.2
        ga.mutation_rate = 0.01
        ga.tol = 1E-6
        ga.convergence_iters = 10000
        ga.max_time = self.ga_time
        ga.optimize_ga()

        optimized_design_variables = ga.optimized_design_variables
        optimized_design_variables = \
            [bool(y) for y in optimized_design_variables]

        self.objective = 0.0

        self.turbine_x = self.x_locations[optimized_design_variables]
        self.turbine_y = self.y_locations[optimized_design_variables]
        self.nturbs = np.sum(optimized_design_variables)
        self.capacity = self.turbine_capacity * self.nturbs
        self.area = self.full_polygons.area
        self.capacity_density = self.capacity / self.area * 1E3

        self.wind_plant["wind_farm_xCoordinates"] = \
            self.x_locations[optimized_design_variables]
        self.wind_plant["wind_farm_yCoordinates"] = \
            self.y_locations[optimized_design_variables]
        self.wind_plant["system_capacity"] = self.capacity
        self.wind_plant.assign_inputs()
        self.wind_plant.execute()
        self.aep = self.wind_plant.annual_energy()

        self.objective = ga.optimized_function_value

    def place_turbines(self):
        """run all functions to define bespoke wind plant turbine layouts
        """
        self.define_exclusions()
        self.initialize_packing()
        self.optimize()
