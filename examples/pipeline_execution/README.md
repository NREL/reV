# reV Pipeline Execution

This set of example files demonstrates how to run the full reV pipeline using the pipeline manager.

The full pipeline can be executed using the following CLI call:

`rev -c "../config_pipeline.json" pipeline`

Please note that the project directory (for configs, logs, outputs, etc...) in all of the configs is set to: `/scratch/gbuster/rev/test_pipeline/` and should be changed for your run.

## Pipeline Input Requirements

The reV pipeline manager will perform several checks to ensure the following input requirements are satisfied.
These requirements are necessary to track pipeline status and to pipe i/o through the modules.

1. All pipeline modules must have the same output directory.
2. Only one pipeline can be run per output directory.
3. Each module run by the pipeline must have a unique job name.

## Failed Jobs

The pipeline manager will keep a status of jobs that are submitted, running, successful, or failed.
If any jobs fail in a pipeline step, the pipeline will wait until all jobs in that step have completed, then raise a failed message.
Error messages can be found in the stdout/stderr files belonging to the respective failed job(s).
The user can re-submit the full pipeline job and only the jobs that failed will be re-run.
If full modules had previously finished successfully, those modules will be skipped.
