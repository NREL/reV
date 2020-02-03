.. _offshore_wind
reV Offshore Wind Execution
===========================

This example includes configs to run a reV wind supply curve analysis for the full CONUS extent including offshore wind.
A few simplifications are made, most notably that a single turbine is used for calculating generation at all sites.

reV Offshore Module Description
-------------------------------

The pipeline includes the offshore module, which is run after the generation module.
The offshore module aggregates generation data (on the WTK resource grid) to the offshore farm meta data (sparse).
Offshore wind farms are therefore represented by the mean capacity factor and windspeed of their neighboring offshore resource pixels.

Offshore LCOE is calculated using ORCA. A seperate turbine technology input is used with a few simple inputs like system capacity and sub structure type.
Any inputs in the turbine json are overwritten by the site-specific offshore data csv where there are overlapping column names.
A warning is printed if data is overwritten in this way.

Treatment of Offshore Points in Supply Curve
--------------------------------------------

Each offshore wind farm point in the offshore data csv is assigned to its own 600 MW supply curve point.
Offshore wind farms will always be 1-to-1 with supply curve points.

Offshore farms are assigned GID's that are 1e7 plus the wind farm id.
The offshore farm GID's are the same as their respective supply curve points.

.. image:: sc_total_lcoe.png
