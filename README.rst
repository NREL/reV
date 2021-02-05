******************************************************
Welcome to the Renewable Energy Potential (reV) Model!
******************************************************

.. image:: https://github.com/NREL/reV/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/reV/

.. image:: https://github.com/NREL/reV/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/reV/actions?query=workflow%3A%22Pytests%22

.. image:: https://github.com/NREL/reV/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/reV/actions?query=workflow%3A%22Lint+Code+Base%22

.. image:: https://img.shields.io/pypi/pyversions/NREL-reV.svg
    :target: https://pypi.org/project/NREL-reV/

.. image:: https://badge.fury.io/py/NREL-reV.svg
    :target: https://badge.fury.io/py/NREL-reV

.. image:: https://anaconda.org/nrel/nrel-rev/badges/version.svg
    :target: https://anaconda.org/nrel/nrel-rev

.. image:: https://anaconda.org/nrel/nrel-rev/badges/license.svg
    :target: https://anaconda.org/nrel/nrel-rev

.. image:: https://codecov.io/gh/nrel/reV/branch/master/graph/badge.svg?token=U4ZU9F0K0Z
    :target: https://codecov.io/gh/nrel/reV

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/NREL/reV/master
    
.. image:: https://zenodo.org/badge/201343076.svg
   :target: https://zenodo.org/badge/latestdoi/201343076

.. inclusion-intro


Recommended Citation
====================

Please cite both the technical paper and the software with the version and DOI you used:

Maclaurin, Galen J., Nicholas W. Grue, Anthony J. Lopez, and Donna M. Heimiller. 2019. “The Renewable Energy Potential (reV) Model: A Geospatial Platform for Technical Potential and Supply Curve Modeling.” Golden, Colorado, United States: National Renewable Energy Laboratory. NREL/TP-6A20-73067. https://doi.org/10.2172/1563140.

Michael Rossol, Grant Buster, and Mike Bannister. The Renewable Energy Potential Model (reV). https://github.com/NREL/reV (version v0.4.46), 2021. https://doi.org/10.5281/zenodo.4501717.


reV command line tools
======================

- `reV <https://nrel.github.io/reV/reV/reV.cli.html#rev>`_
- `reV-project-points <https://nrel.github.io/reV/reV/reV.config.cli_project_points.html#rev-project-points>`_
- `reV-gen <https://nrel.github.io/reV/reV/reV.generation.cli_gen.html#rev-gen>`_
- `reV-econ <https://nrel.github.io/reV/reV/reV.econ.cli_econ.html#rev-econ>`_
- `reV-offshore <https://nrel.github.io/reV/reV/reV.offshore.cli_offshore.html#rev-offshore>`_
- `reV-collect <https://nrel.github.io/reV/reV/reV.handlers.cli_collect.html#rev-collect>`_
- `reV-multiyear <https://nrel.github.io/reV/reV/reV.handlers.cli_multi_year.html#rev-multiyear>`_
- `reV-supply-curve-aggregation <https://nrel.github.io/reV/reV/reV.supply_curve.cli_sc_aggregation.html#rev-supply-curve-aggregation>`_
- `reV-supply-curve <https://nrel.github.io/reV/reV/reV.supply_curve.cli_supply_curve.html#rev-supply-curve>`_
- `reV-rep-profiles <https://nrel.github.io/reV/reV/reV.rep_profiles.cli_rep_profiles.html#rev-rep-profiles>`_
- `reV-pipeline <https://nrel.github.io/reV/reV/reV.pipeline.cli_pipeline.html#rev-pipeline>`_
- `reV-batch <https://nrel.github.io/reV/reV/reV.batch.cli_batch.html#rev-batch>`_
- `reV-QA-QC <https://nrel.github.io/reV/reV/reV.qa_qc.cli_qa_qc.html#rev-qa-qc>`_

Using Eagle Env
===============

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled
conda env:

.. code-block:: bash

    conda activate /shared-projects/rev/modulefiles/conda_env/

or

.. code-block:: bash

    source activate /shared-projects/rev/modulefiles/conda_env/

.. or module:

.. .. code-block:: bash

..     module use /shared-projects/rev/modulefiles
..     module load reV

.. **NOTE: Loading the reV module can take several minutes**

Launching a run
---------------

Tips

- Only use a screen session if running the pipeline module: `screen -S rev`
- `Full pipeline execution <https://nrel.github.io/reV/misc/examples.full_pipeline_execution.html>`_

.. code-block:: bash

    reV -c "/scratch/user/rev/config_pipeline.json" pipeline

- Running simply generation or econ can just be done from the console:

.. code-block:: bash

    reV -c "/scratch/user/rev/config_pipeline.json" generation

General Run times and Node configuration on Eagle
-------------------------------------------------

- WTK Conus: 10-20 nodes per year walltime 1-4 hours
- NSRDB Conus: 5 nodes walltime 2 hours

`Eagle node requests <https://nrel.github.io/reV/misc/examples.eagle_node_requests.html>`_

Installing reV
==============

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name rev python=3.7``

2. Activate directory:
    ``conda activate rev``

3. Install reV:
    1) ``pip install NREL-reV`` or
    2) ``conda install nrel-rev --channel=nrel``

       - NOTE: If you install using conda and want to use `HSDS <https://github.com/NREL/hsds-examples>`_
         you will also need to install h5pyd manually: ``pip install h5pyd``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone https://github.com/NREL/reV.git``
    1) enter github username
    2) enter github password

2. Create ``reV`` environment and install package
    1) Create a conda env: ``conda create -n rev``
    2) Run the command: ``conda activate rev``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from master!)
    5) Install ``reV`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

3. Check that ``reV`` was installed successfully
    1) From any directory, run the following commands. This should return the
       help pages for the CLI's.

        - ``reV``
