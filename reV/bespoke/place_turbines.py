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
from reV.utilities.exceptions import WhileLoopPackingError


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
        objective_function : str
            The objective function of the optimization as a string, should
            return the objective to be minimized during layout optimization.
            Variables available are:
                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
                - cost: the annual cost of the wind plant (from cost_function)
        cost_function : str
            The cost function as a string, should return the annual cost
            of the wind farm. Variables available are:
                - n_turbines: the number of turbines
                - system_capacity: wind plant capacity
                - aep: annual energy production
                - self.wind_plant: the SAM wind plant object, through which
                all SAM variables can be accessed
        exclusions : ExclusionMaskFromDict
            The exclusions that define where turbines can be placed. Contains
            exclusions.latitude, exclusions.longitude, and exclusions.mask
        min_spacing : float
            The minimum spacing between turbines (in meters).
        ga_time : float
            The time to run the genetic algorithm (in seconds).
        """

        # inputs
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
        self.optimized_design_variables = None

        self.ILLEGAL = ('import ', 'os.', 'sys.', '.__', '__.', 'eval', 'exec')
        self._preflight(self.objective_function)
        self._preflight(self.cost_function)

    def _preflight(self, eqn):
        """Run preflight checks on the equation string."""
        for substr in self.ILLEGAL:
            if substr in str(eqn):
                msg = ('Will not evaluate string which contains "{}": {}'
                       .format(substr, eqn))
                raise ValueError(msg)

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

        if safe_polygons.area == 0.0:
            self.full_polygons = MultiPolygon([])
            self.packing_polygons = MultiPolygon([])
        else:
            minx, miny, maxx, maxy = safe_polygons.bounds
            safe_polygons = shapely.affinity.translate(safe_polygons,
                                                       xoff=-minx,
                                                       yoff=-miny)
            self.full_polygons = safe_polygons.buffer(0)

            # add extra setback to cell boundary
            minx, miny, maxx, maxy = self.full_polygons.bounds
            minx += self.min_spacing / 2.0
            miny += self.min_spacing / 2.0
            maxx -= self.min_spacing / 2.0
            maxy -= self.min_spacing / 2.0

            boundary_poly = \
                Polygon(((minx, miny), (minx, maxy), (maxx, maxy),
                         (maxx, miny)))
            packing_polygons = boundary_poly.intersection(self.full_polygons)
            if isinstance(packing_polygons, MultiPolygon):
                self.packing_polygons = packing_polygons
            elif isinstance(packing_polygons, Polygon):
                self.packing_polygons = MultiPolygon([packing_polygons])
            else:
                self.packing_polygons = MultiPolygon([])

    def initialize_packing(self):
        """run the turbine packing algorithm (maximizing plant capacity) to
        define potential turbine locations that will be used as design
        variables in the gentic algorithm.
        """

        packing = PackTurbines(self.min_spacing, self.packing_polygons)
        nturbs = 1E6
        mult = 1.0
        iters = 0
        while nturbs > 300:
            iters += 1
            if iters > 10000:
                msg = ('Too many attempts within initialize packing')
                raise WhileLoopPackingError(msg)
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
        # pylint: disable=W0641
        aep = self.wind_plant.annual_energy()
        # pylint: disable=W0123,W0641
        cost = eval(self.cost_function, globals(), locals())
        # pylint: disable=W0123
        objective = eval(self.objective_function, globals(), locals())

        return objective

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
                              self.optimization_objective,
                              max_generation=10000, population_size=25,
                              crossover_rate=0.2, mutation_rate=0.01,
                              tol=1E-6, convergence_iters=10000,
                              max_time=self.ga_time)

        ga.optimize_ga()

        optimized_design_variables = ga.optimized_design_variables
        self.optimized_design_variables = \
            [bool(y) for y in optimized_design_variables]

    def place_turbines(self):
        """run all functions to define bespoke wind plant turbine layouts
        """
        self.define_exclusions()
        self.initialize_packing()
        self.optimize()

    @property
    def turbine_x(self):
        """This is the final optimized turbine x locations"""
        if self.optimized_design_variables is not None:
            return self.x_locations[self.optimized_design_variables]
        else:
            return None

    @property
    def turbine_y(self):
        """This is the final optimized turbine y locations"""
        if self.optimized_design_variables is not None:
            return self.y_locations[self.optimized_design_variables]
        else:
            return None

    @property
    def nturbs(self):
        """This is the final optimized number of turbines"""
        if self.optimized_design_variables is not None:
            return np.sum(self.optimized_design_variables)
        else:
            return None

    @property
    def capacity(self):
        """This is the final optimized plant capacity"""
        if self.optimized_design_variables is not None:
            return self.turbine_capacity * self.nturbs
        else:
            return None

    @property
    def area(self):
        """This is the area available for wind turbine placement"""
        if self.full_polygons is not None:
            return self.full_polygons.area
        else:
            return None

    @property
    def capacity_density(self):
        """This is the optimized capacity density of the wind plant
        defined with the area available after removing the exclusions"""
        if self.full_polygons is None or self.capacity is None:
            return None
        else:
            if self.area != 0.0:
                return self.capacity / self.area * 1E3
            else:
                return 0.0

    @property
    def aep(self):
        """This is the annual energy production of the optimized plant"""
        if self.optimized_design_variables is not None:
            self.wind_plant["wind_farm_xCoordinates"] = self.turbine_x
            self.wind_plant["wind_farm_yCoordinates"] = self.turbine_y
            self.wind_plant["system_capacity"] = self.capacity
            self.wind_plant.assign_inputs()
            self.wind_plant.execute()
            return self.wind_plant.annual_energy()
        else:
            return None

    @property
    def annual_cost(self):
        """This is the annual cost of the optimized plant"""
        if self.optimized_design_variables is not None:
            # pylint: disable=W0641
            system_capacity = self.capacity
            # pylint: disable=W0641
            aep = self.aep
            # pylint: disable=W0123
            return eval(self.cost_function, globals(), locals())
        else:
            return None

    @property
    def objective(self):
        """This is the optimized objective function value"""
        if self.optimized_design_variables is not None:
            # pylint: disable=W0641
            system_capacity = self.capacity
            # pylint: disable=W0641
            aep = self.aep
            # pylint: disable=W0641
            cost = self.annual_cost
            # pylint: disable=W0123
            return eval(self.objective_function, globals(), locals())
        else:
            return None
