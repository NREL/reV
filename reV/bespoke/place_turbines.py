# -*- coding: utf-8 -*-
"""
place turbines for bespoke wind plants
"""
import numpy as np

from shapely.geometry import Polygon, MultiPolygon

from reV.bespoke.pack_turbs import PackTurbines
from reV.bespoke.gradient_free import GeneticAlgorithm
from reV.utilities.exceptions import WhileLoopPackingError


class PlaceTurbines():
    """Framework for optimizing turbine locations for site specific
    exclusions, wind resources, and objective
    """

    def __init__(self, wind_plant, objective_function,
                 capital_cost_function,
                 fixed_operating_cost_function,
                 variable_operating_cost_function,
                 include_mask, pixel_side_length, min_spacing,
                 wake_loss_multiplier=1):
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
                - fixed_charge_rate: user input fixed_charge_rate if included
                  as part of the sam system config.
                - capital_cost: plant capital cost as evaluated
                  by `capital_cost_function`
                - fixed_operating_cost: plant fixed annual operating cost as
                  evaluated by `fixed_operating_cost_function`
                - variable_operating_cost: plant variable annual operating cost
                  as evaluated by `variable_operating_cost_function`
                - self.wind_plant: the SAM wind plant object, through which
                  all SAM variables can be accessed
                - cost: the annual cost of the wind plant (from cost_function)
        capital_cost_function : str
            The plant capital cost function as a string, must return the total
            capital cost in $. Has access to the same variables as the
            objective_function.
        fixed_operating_cost_function : str
            The plant annual fixed operating cost function as a string, must
            return the fixed operating cost in $/year. Has access to the same
            variables as the objective_function.
        variable_operating_cost_function : str
            The plant annual variable operating cost function as a string, must
            return the variable operating cost in $/kWh. Has access to the same
            variables as the objective_function.
        exclusions : ExclusionMaskFromDict
            The exclusions that define where turbines can be placed. Contains
            exclusions.latitude, exclusions.longitude, and exclusions.mask
        min_spacing : float
            The minimum spacing between turbines (in meters).
        wake_loss_multiplier : float, optional
            A multiplier used to scale the annual energy lost due to
            wake losses. **IMPORTANT**: This multiplier will ONLY be
            applied during the optimization process and will NOT be
            come through in output values such as aep, any of the cost
            functions, or even the output objective.
        """

        # inputs
        self.wind_plant = wind_plant

        self.capital_cost_function = capital_cost_function
        self.fixed_operating_cost_function = fixed_operating_cost_function
        self.variable_operating_cost_function = \
            variable_operating_cost_function

        self.objective_function = objective_function
        self.include_mask = include_mask
        self.pixel_side_length = pixel_side_length
        self.min_spacing = min_spacing
        self.wake_loss_multiplier = wake_loss_multiplier

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
        self.safe_polygons = None

        self.ILLEGAL = ('import ', 'os.', 'sys.', '.__', '__.', 'eval', 'exec')
        self._preflight(self.objective_function)
        self._preflight(self.capital_cost_function)
        self._preflight(self.fixed_operating_cost_function)
        self._preflight(self.variable_operating_cost_function)

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
        nx, ny = np.shape(self.include_mask)
        self.safe_polygons = MultiPolygon()
        side_x = np.arange(nx + 1) * self.pixel_side_length
        side_y = np.arange(ny + 1, -1, -1) * self.pixel_side_length
        floored = np.floor(self.include_mask)
        for i in range(nx):
            for j in range(ny):
                if floored[j, i] == 1:
                    added_poly = Polygon(((side_x[i], side_y[j]),
                                          (side_x[i + 1], side_y[j]),
                                          (side_x[i + 1], side_y[j + 1]),
                                          (side_x[i], side_y[j + 1])))
                    self.safe_polygons = self.safe_polygons.union(added_poly)

        if self.safe_polygons.area == 0.0:
            self.full_polygons = MultiPolygon([])
            self.packing_polygons = MultiPolygon([])
        else:
            self.full_polygons = self.safe_polygons.buffer(0)

            # add extra setback to cell boundary
            minx = 0.0
            miny = 0.0
            maxx = nx * self.pixel_side_length
            maxy = ny * self.pixel_side_length
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

    # pylint: disable=W0641,W0123
    def optimization_objective(self, x):
        """The optimization objective used in the bespoke optimization
        """
        x = [bool(y) for y in x]
        if len(x) > 0:
            n_turbines = np.sum(x)
            self.wind_plant["wind_farm_xCoordinates"] = self.x_locations[x]
            self.wind_plant["wind_farm_yCoordinates"] = self.y_locations[x]

            system_capacity = n_turbines * self.turbine_capacity
            self.wind_plant["system_capacity"] = system_capacity

            self.wind_plant.assign_inputs()
            self.wind_plant.execute()
            aep = self._aep_after_scaled_wake_losses()
        else:
            n_turbines = system_capacity = aep = 0

        fixed_charge_rate = self.fixed_charge_rate
        capital_cost = eval(self.capital_cost_function,
                            globals(), locals())
        fixed_operating_cost = eval(self.fixed_operating_cost_function,
                                    globals(), locals())
        variable_operating_cost = eval(self.variable_operating_cost_function,
                                       globals(), locals())

        objective = eval(self.objective_function, globals(), locals())

        return objective

    def _aep_after_scaled_wake_losses(self):
        """AEP after scaling the energy lost due to wake."""
        wake_loss_pct = self.wind_plant['wake_losses']
        aep = self.wind_plant['annual_energy']
        agep = self.wind_plant['annual_gross_energy']

        energy_lost_due_to_wake = wake_loss_pct / 100 * agep
        aep_after_wake_losses = agep - energy_lost_due_to_wake
        other_losses_multiplier = 1 - aep / aep_after_wake_losses

        scaled_wake_losses = (self.wake_loss_multiplier
                              * energy_lost_due_to_wake)
        aep_after_scaled_wake_losses = max(0, agep - scaled_wake_losses)
        return aep_after_scaled_wake_losses * (1 - other_losses_multiplier)

    def optimize(self, **kwargs):
        """Optimize wind farm layout.

        Use a genetic algorithm to optimize wind plant layout for the
        user-defined objective function.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to GA initialization.

        See Also
        --------
        :class:`~reV.bespoke.gradient_free.GeneticAlgorithm` : GA Algorithm.
        """
        nlocs = len(self.x_locations)
        bits = np.ones(nlocs, dtype=int)
        bounds = np.zeros((nlocs, 2), dtype=int)
        bounds[:, 1] = 2
        variable_type = np.array([])
        for _ in range(nlocs):
            variable_type = np.append(variable_type, "int")

        ga_kwargs = {
            'max_generation': 10000,
            'population_size': 25,
            'crossover_rate': 0.2,
            'mutation_rate': 0.01,
            'tol': 1E-6,
            'convergence_iters': 10000,
            'max_time': 3600
        }

        ga_kwargs.update(kwargs)

        ga = GeneticAlgorithm(bits, bounds, variable_type,
                              self.optimization_objective,
                              **ga_kwargs)

        ga.optimize_ga()

        optimized_design_variables = ga.optimized_design_variables
        self.optimized_design_variables = \
            [bool(y) for y in optimized_design_variables]

        self.wind_plant["wind_farm_xCoordinates"] = self.turbine_x
        self.wind_plant["wind_farm_yCoordinates"] = self.turbine_y
        self.wind_plant["system_capacity"] = self.capacity

    def place_turbines(self, **kwargs):
        """Define bespoke wind plant turbine layouts.

        Run all functions to define bespoke wind plant turbine layouts.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to GA initialization.

        See Also
        --------
        :class:`~reV.bespoke.gradient_free.GeneticAlgorithm` : GA Algorithm.
        """
        self.define_exclusions()
        self.initialize_packing()
        self.optimize(**kwargs)

    @property
    def turbine_x(self):
        """This is the final optimized turbine x locations (m)"""
        if self.optimized_design_variables is not None:
            return self.x_locations[self.optimized_design_variables]
        else:
            return None

    @property
    def turbine_y(self):
        """This is the final optimized turbine y locations (m)"""
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
        """This is the final optimized plant nameplate capacity (kW)"""
        if self.optimized_design_variables is not None:
            return self.turbine_capacity * self.nturbs
        else:
            return None

    @property
    def area(self):
        """This is the area available for wind turbine placement (km2)"""
        if self.full_polygons is not None:
            return self.full_polygons.area
        else:
            return None

    @property
    def fixed_charge_rate(self):
        """Fixed charge rate if input to the SAM WindPowerPD object, None if
        not found in inputs."""
        return self.wind_plant.sam_sys_inputs.get('fixed_charge_rate', None)

    @property
    def capacity_density(self):
        """This is the optimized capacity density of the wind plant
        defined with the area available after removing the exclusions
        (MW/km2)"""
        if self.full_polygons is None or self.capacity is None:
            return None
        else:
            if self.area != 0.0:
                return self.capacity / self.area * 1E3
            else:
                return 0.0

    @property
    def aep(self):
        """This is the annual energy production of the optimized plant (kWh)"""
        if self.optimized_design_variables is not None:
            if self.nturbs > 0:
                self.wind_plant["wind_farm_xCoordinates"] = self.turbine_x
                self.wind_plant["wind_farm_yCoordinates"] = self.turbine_y
                self.wind_plant["system_capacity"] = self.capacity
                self.wind_plant.assign_inputs()
                self.wind_plant.execute()
                return self.wind_plant.annual_energy()
            else:
                return 0
        else:
            return None

    # pylint: disable=W0641,W0123
    @property
    def capital_cost(self):
        """This is the capital cost of the optimized plant ($)"""
        if self.optimized_design_variables is not None:
            fixed_charge_rate = self.fixed_charge_rate
            n_turbines = self.nturbs
            system_capacity = self.capacity
            aep = self.aep
            return eval(self.capital_cost_function, globals(), locals())
        else:
            return None

    # pylint: disable=W0641,W0123
    @property
    def fixed_operating_cost(self):
        """This is the annual fixed operating cost of the
        optimized plant ($/year)"""
        if self.optimized_design_variables is not None:
            fixed_charge_rate = self.fixed_charge_rate
            n_turbines = self.nturbs
            system_capacity = self.capacity
            aep = self.aep
            return eval(self.fixed_operating_cost_function,
                        globals(), locals())
        else:
            return None

    # pylint: disable=W0641,W0123
    @property
    def variable_operating_cost(self):
        """This is the annual variable operating cost of the
        optimized plant ($/kWh)"""
        if self.optimized_design_variables is not None:
            fixed_charge_rate = self.fixed_charge_rate
            n_turbines = self.nturbs
            system_capacity = self.capacity
            aep = self.aep
            return eval(self.variable_operating_cost_function,
                        globals(), locals())
        else:
            return None

    # pylint: disable=W0641,W0123
    @property
    def objective(self):
        """This is the optimized objective function value"""
        if self.optimized_design_variables is not None:
            fixed_charge_rate = self.fixed_charge_rate
            n_turbines = self.nturbs
            system_capacity = self.capacity
            aep = self.aep
            capital_cost = self.capital_cost
            fixed_operating_cost = self.fixed_operating_cost
            variable_operating_cost = self.variable_operating_cost
            return eval(self.objective_function, globals(), locals())
        else:
            return None
