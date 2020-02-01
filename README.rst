reV
###
The Renewable Energy Potential (reV) Model

reV command line tools
***********************

- `reV <https://nrel.github.io/reV/reV/reV.cli.html#rev>`_
- `reV-gen <https://nrel.github.io/reV/reV/reV.generation.cli_gen.html#rev-gen>`_
- `reV-econ <https://nrel.github.io/reV/reV/reV.econ.cli_econ.html#rev-econ>`_
- `reV-offshore <https://nrel.github.io/reV/reV/reV.offshore.cli_offshore.html#rev-offshore>`_
- `reV-collect <https://nrel.github.io/reV/reV/reV.handlers.cli_collect.html#rev-collect>`_
- `reV-multiyear <https://nrel.github.io/reV/reV/reV.handlers.cli_multi_year.html#rev-multiyear>`_
- `reV-aggregation <https://nrel.github.io/reV/reV/reV.supply_curve.cli_aggregation.html#rev-aggregation>`_
- `reV-supply-curve <https://nrel.github.io/reV/reV/reV.supply_curve.cli_supply_curve.html#rev-supply-curve>`_
- `reV-rep-profiles <https://nrel.github.io/reV/reV/reV.rep_profiles.cli_rep_profiles.html#rev-rep-profiles>`_
- `reV-pipeline <https://nrel.github.io/reV/reV/reV.pipeline.cli_pipeline.html#rev-pipeline>`_
- `reV-batch <https://nrel.github.io/reV/reV/reV.batch.cli_batch.html#rev-batch>`_

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

- `Full pipeline example <https://github.com/NREL/reV/tree/master/examples/full_pipeline_execution>`_

General Run times and Node configuration on Eagle
=================================================

- WTK Conus: 10-20 nodes per year walltime 1-4 hours
- NSRDB Conus: 5 nodes walltime 2 hours

Execution examples: :ref:`examples`

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
