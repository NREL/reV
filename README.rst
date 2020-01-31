reV
###
The Renewable Energy Potential (reV) Model

reV command line tools
***********************

- `reV <https://github.com/NREL/reV/tree/master/examples/single_module_execution/README.rst>`_
- ``reV-gen``
- `reV-econ <https://github.com/NREL/reV/tree/master/examples/advanced_econ_modeling/README.rst>`_
- `reV-offshore <https://github.com/NREL/reV/tree/master/examples/offshore_wind/README.rst>`_
- ``reV-collect``
- ``reV-multiyear``
- ``reV-aggregation``
- ``reV-supply-curve``
- ``reV-rep-profiles``
- `reV-pipeline <https://github.com/NREL/reV/tree/master/examples/full_pipeline_execution/README.rst>`_
- `reV-batch <https://github.com/NREL/reV/tree/master/examples/batched_execution/README.rst>`_

Using Eagle Module
******************

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled module:
::
    module use /shared-projects/rev/modulefiles
    module load reV

Launching a run
===============

Tips

- only use a screen session if running the pipeline module: `screen -S rev`
- Running simply generation or lcoe can just be done from the console:

    reV -c "/scratch/user/rev/config_pipeline.json" pipeline

- `Full pipeline example here <https://github.com/NREL/reV/tree/master/examples/full_pipeline_execution>`_

General Run times and Node configuration on Eagle
=================================================

- WTK Conus: 10-20 nodes per year walltime 1-4 hours
- NSRDB Conus: 5 nodes walltime 2 hours

`Execution examples. <https://github.com/NREL/reV/tree/master/examples>`_

Installing reV
**************

Option 1: PIP Install the most recent version of master (recommended for analysts):
===================================================================================

1. Create a new environment: ``conda create --name rev python=3.7``
2. Activate directory: ``conda activate rev``
3. Install reV: ``pip install git+ssh://git@github.com/NREL/reV.git`` or ``pip install git+https://github.com/NREL/reV.git``

Option 2: Clone repo (recommended for developers)
=================================================

1. from home dir, ``git clone https://github.com/NREL/reV2.git``
    1) enter github username
    2) enter github password

2. Install reV environment and modules (using conda)
    1) cd into reV repo cloned above
    2) cd into ``bin/$OS/``
    3) run the command: ``conda env create -f rev.yml``. If conda can't find any packages, try removing them from the yml file.
    4) run the command: ``conda activate rev``
    5) prior to running ``pip`` below, make sure branch is correct (install from master!)
    6) cd back to the reV repo (where setup.py is located)
    7) install pre-commit: ``pre-commit install``
    8) run ``pip install .`` (or ``pip install -e .`` if running a dev branch or working on the source code)

3. Check that rev was installed successfully
    1) From any directory, run the following commands. This should return the help pages for the CLI's.
        - ``reV``
