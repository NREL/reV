reV Parameter Batching
######################

This example set shows how reV inputs can be parameterized and the execution can be batched.

Batching Config Description
***************************
 - "sets" in the batch config is a list of batches.
 - Each "batch" is a dictionary containing "args" and "files".
 - "args" are the key/value pairs from which the batching combinations will be made. Each unique combination of args represents a job. If two args are specified with three possible values each, nine jobs will be run.
 - The unique combinations of "args" will be replaced in all files in the "files" list. The arg must already exist in the file for the new values to be inserted. The replacement is done recursively.
 - Batch jobs will be assigned names based on the args. Accordingly, the name field specification should be omitted in all configs.

How to Run
**********
All batch jobs will be kicked off using the following CLI call:

::
    rev -c "../config_batch.json" batch

New sub directories will be created in the folder with the batch config file for each sub job.
All job files in the same directory (and sub directories) as the batch config file will be copied into the job folders.
The reV pipeline manager will be executed in each sub directory.
The above batch cli command can be issues repeatedly to clean up the sub directory status ``.jsons``, kick off the next step in the pipeline, or to rerun failed jobs.
See the reV pipeline execution example for more details on how the pipeline works.

All of the batch jobs can be collected into a single file using the multi-year collection utility.
This utility is not part of the batch pipeline and needs to be executed and configured separately.
See the ``config_multi-year.json`` file for details on how to setup this collection step.
To execute, use the following command:

::
    rev -c "../config_multi-year.json" multi-year
