# reV2 Pipeline Execution

This set of example files demonstrates how to run the full reV2 pipeline using the pipeline manager.

The full pipeline can be executed using the following CLI call:

`python cli.py -c "/scratch/gbuster/rev/test_pipeline/config_pipeline.json" pipeline`

Please note that the project directory (for configs, logs, outputs, etc...) in all of the configs is set to: `/scratch/gbuster/rev/test_pipeline/` and should be changed for your run. 