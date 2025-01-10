.. raw:: html

    <p align="center">
        <img height="180" src="docs/source/_static/logo.png" />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img height="170" src="docs/source/_static/RD100_2023_Winner_Logo.png" />
    </p>

---------

|Docs| |Tests| |Linter| |PythonV| |Pypi| |Codecov| |Zenodo| |Binder|

.. |Docs| image:: https://github.com/NREL/reV/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/reV/

.. |Tests| image:: https://github.com/NREL/reV/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/reV/actions?query=workflow%3A%22Pytests%22

.. |Linter| image:: https://github.com/NREL/reV/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/reV/actions?query=workflow%3A%22Lint+Code+Base%22

.. |PythonV| image:: https://img.shields.io/pypi/pyversions/NREL-reV.svg
    :target: https://pypi.org/project/NREL-reV/

.. |Pypi| image:: https://badge.fury.io/py/NREL-reV.svg
    :target: https://badge.fury.io/py/NREL-reV

.. |Codecov| image:: https://codecov.io/gh/nrel/reV/branch/main/graph/badge.svg?token=U4ZU9F0K0Z
    :target: https://codecov.io/gh/nrel/reV

.. |Zenodo| image:: https://zenodo.org/badge/201343076.svg
   :target: https://zenodo.org/badge/latestdoi/201343076

.. |Binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/nrel/reV/HEAD

|

.. inclusion-intro

**reV** (the Renewable Energy Potential model)
is an open-source geospatial techno-economic tool that
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
reV is a set of `Python classes and functions <https://nrel.github.io/reV/_autosummary/reV.html>`_
that can be executed on HPC systems using `CLI commands <https://nrel.github.io/reV/_cli/cli.html>`_.
A full reV execution consists of one or more compute modules
(each consisting of their own Python class/CLI command)
strung together using a `pipeline framework <https://nrel.github.io/reV/_cli/reV%20pipeline.html>`_,
or configured using `batch <https://nrel.github.io/reV/_cli/reV%20batch.html>`_.

A typical reV workflow begins with input wind/solar/geothermal resource data
(following the `rex data format <https://nrel.github.io/rex/misc/examples.nsrdb.html#data-format>`_)
that is passed through the generation module. This output is then collected across space and time
(if executed on the HPC), before being sent off to be aggregated under user-specified land exclusion scenarios.
Exclusion data is typically provided via a collection of high-resolution spatial data layers stored in an HDF5 file.
This file must be readable by reV's
`ExclusionLayers <https://nrel.github.io/reV/_autosummary/reV.handlers.exclusions.ExclusionLayers.html#reV.handlers.exclusions.ExclusionLayers>`_
class. See the `reVX Setbacks utility <https://nrel.github.io/reVX/misc/examples.setbacks.html>`_
for instructions on generating setback exclusions for use in reV.
Next, transmission costs are computed for each aggregated
"supply-curve point" using user-provided transmission cost tables.
See the `reVX transmission cost calculator utility <https://github.com/NREL/reVX/tree/main/reVX/least_cost_xmission/>`_
for instructions on generating transmission cost tables.
Finally, the supply curves and initial generation data can be used to
extract representative generation profiles for each supply curve point.

A visual summary of this process is given below:


.. inclusion-flowchart

.. raw:: html

    <p align="center">
        <img height="400" src="docs/source/_static/rev_flow_chart.png" />
    </p>

|

.. inclusion-get-started

To get up and running with reV, first head over to the `installation page <https://nrel.github.io/reV/misc/installation.html>`_,
then check out some of the `Examples <https://nrel.github.io/reV/misc/examples.html>`_ or
go straight to the `CLI Documentation <https://nrel.github.io/reV/_cli/cli.html>`_!
You can also check out the `guide on running GAPs models <https://nrel.github.io/gaps/misc/examples.users.html>`_.

.. inclusion-install


Installing reV
==============

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Install from PIP (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name rev python=3.9``

2. Activate directory:
    ``conda activate rev``

3. Install reV:
    1) ``pip install NREL-reV`` or

       - NOTE: If you install using conda and want to run from files directly on S3 like in the `running reV locally example <https://nrel.github.io/reV/misc/examples.running_locally.html>`_
         you will also need to install S3 filesystem dependencies: ``pip install NREL-reV[s3]``

       - NOTE: If you install using conda and want to use `HSDS <https://github.com/NREL/hsds-examples>`_
         you will also need to install HSDS dependencies: ``pip install NREL-reV[hsds]``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/reV.git``

2. Create ``reV`` environment and install package
    1) Create a conda env: ``conda create -n rev``
    2) Run the command: ``conda activate rev``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``reV`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that ``reV`` was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``reV``


reV command line tools
======================

- `reV <https://nrel.github.io/reV/_cli/reV.html#reV>`_
- `reV template-configs <https://nrel.github.io/reV/_cli/reV%20template-configs.html>`_
- `reV batch <https://nrel.github.io/reV/_cli/reV%20batch.html>`_
- `reV pipeline <https://nrel.github.io/reV/_cli/reV%20pipeline.html>`_
- `reV project-points <https://nrel.github.io/reV/_cli/reV%20project-points.html>`_
- `reV bespoke <https://nrel.github.io/reV/_cli/reV%20bespoke.html>`_
- `reV generation <https://nrel.github.io/reV/_cli/reV%20generation.html>`_
- `reV econ <https://nrel.github.io/reV/_cli/reV%20econ.html>`_
- `reV collect <https://nrel.github.io/reV/_cli/reV%20collect.html>`_
- `reV multiyear <https://nrel.github.io/reV/_cli/reV%20multiyear.html>`_
- `reV supply-curve-aggregation <https://nrel.github.io/reV/_cli/reV%20supply-curve-aggregation.html>`_
- `reV supply-curve <https://nrel.github.io/reV/_cli/reV%20supply-curve.html>`_
- `reV rep-profiles <https://nrel.github.io/reV/_cli/reV%20rep-profiles.html>`_
- `reV hybrids <https://nrel.github.io/reV/_cli/reV%20hybrids.html>`_
- `reV nrwal <https://nrel.github.io/reV/_cli/reV%20nrwal.html>`_
- `reV qa-qc <https://nrel.github.io/reV/_cli/reV%20qa-qc.html>`_
- `reV script <https://nrel.github.io/reV/_cli/reV%20script.html>`_
- `reV status <https://nrel.github.io/reV/_cli/reV%20status.html>`_
- `reV reset-status <https://nrel.github.io/reV/_cli/reV%20reset-status.html>`_


Launching a run
---------------

Tips

- Only use a screen session if running the pipeline module: `screen -S rev`
- `Full pipeline execution <https://nrel.github.io/reV/misc/examples.full_pipeline_execution.html>`_

.. code-block:: bash

    reV pipeline -c "/scratch/user/rev/config_pipeline.json"

- Running simply generation or econ can just be done from the console:

.. code-block:: bash

    reV generation -c "/scratch/user/rev/config_gen.json"

General Run times and Node configuration on Eagle
-------------------------------------------------

- WTK Conus: 10-20 nodes per year walltime 1-4 hours
- NSRDB Conus: 5 nodes walltime 2 hours

`Eagle node requests <https://nrel.github.io/reV/misc/examples.eagle_node_requests.html>`_


.. inclusion-citation


Recommended Citation
====================

Please cite both the technical paper and the software with the version and
DOI you used:

Maclaurin, Galen J., Nicholas W. Grue, Anthony J. Lopez, Donna M. Heimiller,
Michael Rossol, Grant Buster, and Travis Williams. 2019. “The Renewable Energy
Potential (reV) Model: A Geospatial Platform for Technical Potential and Supply
Curve Modeling.” Golden, Colorado, United States: National Renewable Energy
Laboratory. NREL/TP-6A20-73067. https://doi.org/10.2172/1563140.

Grant Buster, Michael Rossol, Paul Pinchuk, Brandon N Benton, Robert Spencer,
Mike Bannister, & Travis Williams. (2023).
NREL/reV: reV 0.8.0 (v0.8.0). Zenodo. https://doi.org/10.5281/zenodo.8247528
