# reV2 Pipeline Execution

This set of example files demonstrates how to run the full reV2 pipeline using the pipeline manager.

The full pipeline can be executed using the following CLI call:

`python cli.py -c "/scratch/gbuster/rev/test_pipeline/config_pipeline.json" pipeline`

## Directory Specifications

Please note that the project directory (for configs, logs, outputs, etc...) in all of the configs is set to: `/scratch/gbuster/rev/test_pipeline/` and should be changed for your run. 

Also note that for the full pipeline to run successfully with pipeline-managed i/o, all modules must have the same output directory, which will also be used as the status directory (where status json files will be saved and parsed). 

## Failed Jobs

The pipeline manager will keep a status of jobs that are submitted, running, successful, or failed. 
If any jobs fail in a pipeline step, the pipeline will terminate. 
Error messages can be found in the stdout/stderr files belonging to the respective failed job(s). 
The user can re-submit the full pipeline job and only the jobs that failed will be re-run. 
If full modules had previously finished successfully, those modules will be skipped. 
