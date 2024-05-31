# -*- coding: utf-8 -*-
# pylint: disable=inconsistent-return-statements
"""
place turbines for bespoke wind plants
"""
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

from reV.bespoke.gradient_free import GeneticAlgorithm
from reV.bespoke.pack_turbs import PackTurbines
from reV.utilities.exceptions import WhileLoopPackingError


def none_until_optimized(func):
    """Decorator that returns None until `PlaceTurbines` is optimized.

    Meant for exclusive use in `PlaceTurbines` and its subclasses.
    `PlaceTurbines` is considered optimized when its
    `optimized_design_variables` attribute is not `None`.

    Parameters
    ----------
    func : callable
        A callable function that should return `None` until
        `PlaceTurbines` is optimized.

    Returns
    -------
    callable
        New function that returns `None` until `PlaceTurbines` is
        optimized.
    """

    def _func(pt):
        """Wrapper to return `None` if `PlaceTurbines` is not optimized"""
        if pt.optimized_design_variables is None:
            return
        return func(pt)
    return _func


class PlaceTurbines:
    """Framework for optimizing turbine locations for site specific
    exclusions, wind resources, and objective
    """

    def __init__(self, wind_plant, objective_function,
                 capital_cost_function,
                 fixed_operating_cost_function,
                 variable_operating_cost_function,
                 balance_of_system_cost_function,
                 include_mask, pixel_side_length, min_spacing):
        """
        Parameters
        ----------
        wind_plant : WindPowerPD
            wind plant object to analyze wind plant performance. This
            object should have everything in the plant defined, such
            that only the turbine coordinates and plant capacity need to
            be defined during the optimization.
        objective_function : str
            The objective function of the optimization as a string,
            should return the objective to be minimized during layout
            optimization. Variables available are:

                - ``n_turbines``: the number of turbines
                - ``system_capacity``: wind plant capacity
                - ``aep``: annual energy production
                - ``avg_sl_dist_to_center_m``: Average straight-line
                  distance to the supply curve point center from all
                  turbine locations (in m). Useful for computing plant
                  BOS costs.
                - ``avg_sl_dist_to_medoid_m``: Average straight-line
                  distance to the medoid of all turbine locations
                  (in m). Useful for computing plant BOS costs.
                - ``nn_conn_dist_m``: Total BOS connection distance
                  using nearest-neighbor connections. This variable is
                  only available for the
                  ``balance_of_system_cost_function`` equation.
                - ``fixed_charge_rate``: user input fixed_charge_rate if
                  included as part of the sam system config.
                - ``capital_cost``: plant capital cost as evaluated
                  by `capital_cost_function`
                - ``fixed_operating_cost``: plant fixed annual operating
                  cost as evaluated by `fixed_operating_cost_function`
                - ``variable_operating_cost``: plant variable annual
                  operating cost as evaluated by
                  `variable_operating_cost_function`
                - ``balance_of_system_cost``: plant balance of system
                  cost as evaluated by `balance_of_system_cost_function`
                - ``self.wind_plant``: the SAM wind plant object,
                  through which all SAM variables can be accessed

        capital_cost_function : str
            The plant capital cost function as a string, must return the
            total capital cost in $. Has access to the same variables as
            the objective_function.
        fixed_operating_cost_function : str
            The plant annual fixed operating cost function as a string,
            must return the fixed operating cost in $/year. Has access
            to the same variables as the objective_function.
        variable_operating_cost_function : str
            The plant annual variable operating cost function as a
            string, must return the variable operating cost in $/kWh.
            Has access to the same variables as the objective_function.
            You can set this to "0" to effectively ignore variable
            operating costs.
        balance_of_system_cost_function : str
            The plant balance-of-system cost function as a string, must
            return the variable operating cost in $. Has access to the
            same variables as the objective_function. You can set this
            to "0" to effectively ignore balance-of-system costs.
        include_mask : np.ndarray
            Supply curve point 2D inclusion mask where included pixels
            are set to 1 and excluded pixels are set to 0.
        pixel_side_length : int
            Side length (m) of a single pixel of the `include_mask`.
        min_spacing : float
            The minimum spacing between turbines (in meters).
        """

        # inputs
        self.wind_plant = wind_plant

        self.capital_cost_function = capital_cost_function
        self.fixed_operating_cost_function = fixed_operating_cost_function
        self.variable_operating_cost_function = \
            variable_operating_cost_function
        self.balance_of_system_cost_function = balance_of_system_cost_function

        self.objective_function = objective_function
        self.include_mask = include_mask
        self.pixel_side_length = pixel_side_length
        self.min_spacing = min_spacing

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
        self._optimized_nn_conn_dist_m = None

        self.ILLEGAL = ('import ', 'os.', 'sys.', '.__', '__.', 'eval', 'exec')
        self._preflight(self.objective_function)
        self._preflight(self.capital_cost_function)
        self._preflight(self.fixed_operating_cost_function)
        self._preflight(self.variable_operating_cost_function)
        self._preflight(self.balance_of_system_cost_function)

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
        ny, nx = np.shape(self.include_mask)
        self.safe_polygons = MultiPolygon()
        side_x = np.arange(nx + 1) * self.pixel_side_length
        side_y = np.arange(ny, -1, -1) * self.pixel_side_length
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
        """Run the turbine packing algorithm (maximizing plant capacity) to
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

    def _sc_center(self):
        """Supply curve point center. """
        ny, nx = np.shape(self.include_mask)
        cx = nx * self.pixel_side_length / 2
        cy = ny * self.pixel_side_length / 2
        return cx, cy

    def _avg_sl_dist_to_cent(self, x_locs, y_locs):
        """Average straight-line distance to center from turb locations. """
        cx, cy = self._sc_center()
        return np.hypot(x_locs - cx, y_locs - cy).mean()

    def _avg_sl_dist_to_med(self, x_locs, y_locs):
        """Average straight-line distance to turbine medoid. """
        cx, cy = _turb_medoid(x_locs, y_locs)
        return np.hypot(x_locs - cx, y_locs - cy).mean()

    # pylint: disable=W0641,W0123
    def optimization_objective(self, x):
        """The optimization objective used in the bespoke optimization
        """
        x = [bool(y) for y in x]
        if len(x) > 0:
            n_turbines = np.sum(x)
            x_locs, y_locs = self.x_locations[x], self.y_locations[x]
            self.wind_plant["wind_farm_xCoordinates"] = x_locs
            self.wind_plant["wind_farm_yCoordinates"] = y_locs

            system_capacity = n_turbines * self.turbine_capacity
            self.wind_plant["system_capacity"] = system_capacity

            self.wind_plant.assign_inputs()
            self.wind_plant.execute()
            aep = self.wind_plant['annual_energy']
            avg_sl_dist_to_center_m = self._avg_sl_dist_to_cent(x_locs, y_locs)
            avg_sl_dist_to_medoid_m = self._avg_sl_dist_to_med(x_locs, y_locs)
            if "nn_conn_dist_m" in self.balance_of_system_cost_function:
                nn_conn_dist_m = _compute_nn_conn_dist(x_locs, y_locs)
        else:
            n_turbines = system_capacity = aep = 0
            avg_sl_dist_to_center_m = avg_sl_dist_to_medoid_m = 0
            nn_conn_dist_m = 0

        fixed_charge_rate = self.fixed_charge_rate
        capital_cost = eval(self.capital_cost_function,
                            globals(), locals())
        fixed_operating_cost = eval(self.fixed_operating_cost_function,
                                    globals(), locals())
        variable_operating_cost = eval(self.variable_operating_cost_function,
                                       globals(), locals())
        balance_of_system_cost = eval(self.balance_of_system_cost_function,
                                      globals(), locals())
        capital_cost *= self.wind_plant.sam_sys_inputs.get(
            'capital_cost_multiplier', 1)
        fixed_operating_cost *= self.wind_plant.sam_sys_inputs.get(
            'fixed_operating_cost_multiplier', 1)
        variable_operating_cost *= self.wind_plant.sam_sys_inputs.get(
            'variable_operating_cost_multiplier', 1)
        balance_of_system_cost *= self.wind_plant.sam_sys_inputs.get(
            'balance_of_system_cost_multiplier', 1)

        objective = eval(self.objective_function, globals(), locals())

        return objective

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

    def capital_cost_per_kw(self, capacity_mw):
        """Capital cost function ($ per kW) evaluated for a given capacity.

        The capacity will be adjusted to be an exact multiple of the
        turbine rating in order to yield an integer number of
        turbines.

        Parameters
        ----------
        capacity_mw : float
            The desired capacity (MW) to sample the cost curve at. Note
            as mentioned above, the capacity will be adjusted to be an
            exact multiple of the turbine rating in order to yield an
            integer number of turbines. For best results, set this
            value to be an integer multiple of the turbine rating.

        Returns
        -------
        capital_cost : float
            Capital cost ($ per kW) for the (adjusted) plant capacity.
        """

        fixed_charge_rate = self.fixed_charge_rate
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        n_turbines = int(round(capacity_mw * 1e3 / self.turbine_capacity))
        system_capacity = n_turbines * self.turbine_capacity
        mult = self.wind_plant.sam_sys_inputs.get(
            'capital_cost_multiplier', 1) / system_capacity
        return eval(self.capital_cost_function, globals(), locals()) * mult

    @property
    def fixed_charge_rate(self):
        """Fixed charge rate if input to the SAM WindPowerPD object, None if
        not found in inputs."""
        return self.wind_plant.sam_sys_inputs.get("fixed_charge_rate", None)

    @property
    @none_until_optimized
    def turbine_x(self):
        """This is the final optimized turbine x locations (m)"""
        return self.x_locations[self.optimized_design_variables]

    @property
    @none_until_optimized
    def turbine_y(self):
        """This is the final optimized turbine y locations (m)"""
        return self.y_locations[self.optimized_design_variables]

    @property
    @none_until_optimized
    def avg_sl_dist_to_center_m(self):
        """This is the final avg straight line distance to center (m)"""
        return self._avg_sl_dist_to_cent(self.turbine_x, self.turbine_y)

    @property
    @none_until_optimized
    def avg_sl_dist_to_medoid_m(self):
        """This is the final avg straight line distance to turb medoid (m)"""
        return self._avg_sl_dist_to_med(self.turbine_x, self.turbine_y)

    @property
    @none_until_optimized
    def nn_conn_dist_m(self):
        """This is the final avg straight line distance to turb medoid (m)"""
        if self._optimized_nn_conn_dist_m is None:
            self._optimized_nn_conn_dist_m = _compute_nn_conn_dist(
                self.turbine_x, self.turbine_y
            )
        return self._optimized_nn_conn_dist_m

    @property
    @none_until_optimized
    def nturbs(self):
        """This is the final optimized number of turbines"""
        return np.sum(self.optimized_design_variables)

    @property
    @none_until_optimized
    def capacity(self):
        """This is the final optimized plant nameplate capacity (kW)"""
        return self.turbine_capacity * self.nturbs

    @property
    @none_until_optimized
    def convex_hull(self):
        """This is the convex hull of the turbine locations"""
        turbines = MultiPoint([Point(x, y)
                               for x, y in zip(self.turbine_x,
                                               self.turbine_y)])
        return turbines.convex_hull

    @property
    @none_until_optimized
    def area(self):
        """This is the area available for wind turbine placement (km^2)"""
        return self.full_polygons.area / 1e6

    @property
    @none_until_optimized
    def convex_hull_area(self):
        """This is the area of the convex hull of the turbines (km^2)"""
        return self.convex_hull.area / 1e6

    @property
    @none_until_optimized
    def full_cell_area(self):
        """This is the full non-excluded area available for wind turbine
        placement (km^2)"""
        nx, ny = np.shape(self.include_mask)
        side_x = nx * self.pixel_side_length
        side_y = ny * self.pixel_side_length
        return side_x * side_y / 1e6

    @property
    @none_until_optimized
    def capacity_density(self):
        """This is the optimized capacity density of the wind plant
        defined with the area available after removing the exclusions
        (MW/km2)"""
        if self.full_polygons is None or self.capacity is None:
            return

        if self.area != 0.0:
            return self.capacity / self.area / 1E3

        return 0.0

    @property
    @none_until_optimized
    def convex_hull_capacity_density(self):
        """This is the optimized capacity density of the wind plant
        defined with the convex hull area of the turbine layout (MW/km2)"""
        if self.convex_hull_area != 0.0:
            return self.capacity / self.convex_hull_area / 1E3
        return 0.0

    @property
    @none_until_optimized
    def full_cell_capacity_density(self):
        """This is the optimized capacity density of the wind plant
        defined with the full non-excluded area of the turbine layout (MW/km2)
        """
        if self.full_cell_area != 0.0:
            return self.capacity / self.full_cell_area / 1E3
        return 0.0

    @property
    @none_until_optimized
    def aep(self):
        """This is the annual energy production of the optimized plant (kWh)"""
        if self.nturbs <= 0:
            return 0

        self.wind_plant["wind_farm_xCoordinates"] = self.turbine_x
        self.wind_plant["wind_farm_yCoordinates"] = self.turbine_y
        self.wind_plant["system_capacity"] = self.capacity
        self.wind_plant.assign_inputs()
        self.wind_plant.execute()
        return self.wind_plant.annual_energy()

    # pylint: disable=W0641,W0123
    @property
    @none_until_optimized
    def capital_cost(self):
        """This is the capital cost of the optimized plant ($)"""
        fixed_charge_rate = self.fixed_charge_rate
        n_turbines = self.nturbs
        system_capacity = self.capacity
        aep = self.aep
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        avg_sl_dist_to_medoid_m = self.avg_sl_dist_to_medoid_m
        nn_conn_dist_m = self.nn_conn_dist_m

        mult = self.wind_plant.sam_sys_inputs.get(
            'capital_cost_multiplier', 1)
        return eval(self.capital_cost_function, globals(), locals()) * mult

    # pylint: disable=W0641,W0123
    @property
    @none_until_optimized
    def fixed_operating_cost(self):
        """This is the annual fixed operating cost of the
        optimized plant ($/year)"""
        fixed_charge_rate = self.fixed_charge_rate
        n_turbines = self.nturbs
        system_capacity = self.capacity
        aep = self.aep
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        avg_sl_dist_to_medoid_m = self.avg_sl_dist_to_medoid_m
        nn_conn_dist_m = self.nn_conn_dist_m

        mult = self.wind_plant.sam_sys_inputs.get(
            'fixed_operating_cost_multiplier', 1)
        return eval(self.fixed_operating_cost_function,
                    globals(), locals()) * mult

    # pylint: disable=W0641,W0123
    @property
    @none_until_optimized
    def variable_operating_cost(self):
        """This is the annual variable operating cost of the
        optimized plant ($/kWh)"""
        fixed_charge_rate = self.fixed_charge_rate
        n_turbines = self.nturbs
        system_capacity = self.capacity
        aep = self.aep
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        avg_sl_dist_to_medoid_m = self.avg_sl_dist_to_medoid_m
        nn_conn_dist_m = self.nn_conn_dist_m

        mult = self.wind_plant.sam_sys_inputs.get(
            'variable_operating_cost_multiplier', 1)
        return eval(self.variable_operating_cost_function,
                    globals(), locals()) * mult

    @property
    @none_until_optimized
    def balance_of_system_cost(self):
        """This is the balance of system cost of the optimized plant ($)"""
        fixed_charge_rate = self.fixed_charge_rate
        n_turbines = self.nturbs
        system_capacity = self.capacity
        aep = self.aep
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        avg_sl_dist_to_medoid_m = self.avg_sl_dist_to_medoid_m
        nn_conn_dist_m = self.nn_conn_dist_m

        mult = self.wind_plant.sam_sys_inputs.get(
            'balance_of_system_cost_multiplier', 1)
        return eval(self.balance_of_system_cost_function,
                    globals(), locals()) * mult

    # pylint: disable=W0641,W0123
    @property
    @none_until_optimized
    def objective(self):
        """This is the optimized objective function value"""
        fixed_charge_rate = self.fixed_charge_rate
        n_turbines = self.nturbs
        system_capacity = self.capacity
        aep = self.aep
        capital_cost = self.capital_cost
        fixed_operating_cost = self.fixed_operating_cost
        variable_operating_cost = self.variable_operating_cost
        balance_of_system_cost = self.balance_of_system_cost
        avg_sl_dist_to_center_m = self.avg_sl_dist_to_center_m
        avg_sl_dist_to_medoid_m = self.avg_sl_dist_to_medoid_m
        nn_conn_dist_m = self.nn_conn_dist_m

        return eval(self.objective_function, globals(), locals())


def _turb_medoid(x_locs, y_locs):
    """Turbine medoid. """
    return np.median(x_locs), np.median(y_locs)


def _compute_nn_conn_dist(x_coords, y_coords):
    """Connect turbines using a greedy nearest-neighbor approach. """
    if len(x_coords) <= 1:
        return 0

    coordinates = np.c_[x_coords, y_coords]
    allowed_conns = np.r_[coordinates.mean(axis=0)[None], coordinates]

    mask = np.zeros_like(allowed_conns)
    mask[0] = 1
    left_to_connect = np.ma.array(allowed_conns, mask=mask)

    mask = np.ones_like(allowed_conns)
    mask[0] = 0
    allowed_conns = np.ma.array(allowed_conns, mask=mask)

    total_dist = 0
    for __ in range(len(coordinates)):
        dists = left_to_connect[:, :, None] - allowed_conns.T[None]
        dists = np.hypot(dists[:, 0], dists[:, 1])
        min_dists = dists.min(axis=-1)
        total_dist += min_dists.min()
        next_connection = min_dists.argmin()
        allowed_conns.mask[next_connection] = 0
        left_to_connect.mask[next_connection] = 1

    return total_dist
