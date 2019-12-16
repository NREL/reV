reV
###
The Renewable Energy Potential (reV) Model

Using Eagle Module
******************

If you would like to run reV on Eagle (NREL's HPC) you can use a pre-compiled module:
::
    module use /projects/rev/modulefiles
    module load reV

Launching a run
===============

Tips

- only use a screen session if running the pipeline module: `screen -S rev`
- Running simply generation or lcoe can just be done from the console:
.. code-block::
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
3. Install reV2: ``pip install git+ssh://git@github.com/NREL/reV.git`` or ``pip install git+https://github.com/NREL/reV.git``

Option 2: Clone repo (recommended for developers)
=================================================

1. from home dir, ``git clone https://github.com/NREL/reV2.git``
    1) enter github username
    2) enter github password

2. Install reV environment and modules (using conda)
    1) cd into reV repo cloned above
    2) cd into ``bin/``
    3) run the command: ``conda env create -f rev.yml``
    4) run the command: ``conda activate rev``
    5) prior to running ``pip`` below, make sure branch is correct (install from master!)
    6) cd back to the reV repo (where setup.py is located)
    7) install pre-commit: ``pre-commit install``
    8) run ``pip install .`` (or ``pip install -e .`` if running a dev branch or working on the source code)

3. Check that rev was installed successfully
    1) From any directory, run the following commands. This should return the help pages for the CLI's.
        - ``reV``
        - ``reV_gen``
