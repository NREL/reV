.. toctree::
   :hidden:

   Home page <self>
   Installation and Usage <misc/installation_usage>
   Examples <misc/examples>
   API reference <_autosummary/reV>
   CLI reference <_cli/cli>

reV documentation
*****************

What is reV?
============
reV stands for **Renewable Energy Potential(V)** model.

The reV model is an open-source geospatial techno-economic tool that
estimates renewable energy technical potential (capacity and generation),
system cost, and supply curves for solar photovoltaics (PV),
concentrating solar power (CSP), geothermal, and wind energy.
reV allows researchers to include exhaustive spatial representation
of the built and natural environment into the generation and cost estimates
that it computes.

reV is highly dynamic, allowing analysts to assess potential at varying levels
of detail — from a single site up to an entire continent at temporal resolutions
ranging from five minutes to hourly, spanning a single year or multiple decades.
The reV model can (and has been used to) provide broad coverage across large spatial
extents, including North America, South and Central Asia, the Middle East, South America,
and South Africa to inform national and international-scale analyses. Still, reV is
equally well-suited for regional infrastructure and deployment planning and analysis.


For a detailed description of reV capabilities and functionality, see the
`NREL reV technical report <https://www.nrel.gov/docs/fy19osti/73067.pdf>`_.

How does reV work?
==================
reV is a set of `python classes and functions <https://nrel.github.io/reV/_autosummary/reV.html>`_
that can be executed on HPC systems using :ref:`CLI commands <cli-docs>`.
A full reV executions consists of one or more compute modules
(each consisting of their own python class/CLI command)
strung together using a :ref:`pipeline framework <rev-pipeline>` or
configured using :ref:`batch <rev-batch>`.

A typical reV workflow begins with input wind/solar/geothermal resource data
(following the `rex data format <https://nrel.github.io/rex/misc/examples.nsrdb.html#data-format>`_)
that is passed through the generation module. This output is then collected
across space and time (if executed on teh HPC), before being sent off to
be aggregated under user-specified land exclusion scenarios. Exclusion data
is typically provided via a collection of high-resolution spatial data layers
stored in an HDF5 file. This file must be readable by reV's
`ExclusionLayers <https://nrel.github.io/reV/_autosummary/reV.handlers.exclusions.ExclusionLayers.html#reV.handlers.exclusions.ExclusionLayers>`_
class. See the `reVX Setbacks utility <https://nrel.github.io/reVX/misc/examples.setbacks.html>`_
for instructions on generating setback exclusions for use in reV.
Next, transmission costs are computed for each aggregated "supply-curve point"
using user-provided transmission cost tables. See the See the
`reVX transmission cost calculator utility <https://github.com/NREL/reVX/tree/main/reVX/least_cost_xmission/>`_
for instructions on generating transmission cost tables. Finally,
the supply curves and initial generation data can be used to extract
representative generation profiles for each supply curve point.

A visual summary of this process is given below:

.. image:: _static/rev_flow_chart.png
  :align: center
  :alt: Typical reV workflow

|

To get up and running with reV, first head over to the :ref:`installation page <installation>`,
then check out some of the :ref:`Examples <examples>` or
go straight to the :ref:`CLI Documentation <cli-docs>`!

|

.. include:: ../../README.rst
   :start-after: inclusion-intro


.. include:: ../../README.rst
   :start-after: inclusion-citation
   :end-before: inclusion-intro
