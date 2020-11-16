Full reV Pipeline Execution
===========================

This set of example files demonstrates how to run the full reV pipeline using
the pipeline manager.

The full pipeline can be executed using the following CLI call:

.. code-block:: bash

    reV -c ./config_pipeline.json pipeline

You can also use the ``--monitor`` flag to continuously monitor the pipeline
and submit jobs for the next pipeline step when the current pipeline step is
complete:

.. code-block:: bash

    reV -c ./config_pipeline.json pipeline --monitor

The continuous monitoring will stop when the full pipeline completes
successfully or if any part of a pipeline step fails. The continuous monitoring
can also be run in a ``nohup`` background process by
adding the ``--background`` flag:

.. code-block:: bash

    reV -c ./config_pipeline.json pipeline --monitor --background

It's important to note that background monitoring will not capture the
stdout/stderr, so you should set the ``log_file`` argument in the pipeline
config json file to log any important messages from the pipeline module.

Finally, if anything goes wrong you can cancel all the pipeline jobs using
the ``--cancel`` flag:

.. code-block:: bash

    reV -c ./config_pipeline.json pipeline --cancel

Pipeline Input Requirements
---------------------------

The reV pipeline manager will perform several checks to ensure the following
input requirements are satisfied. These requirements are necessary to track the
pipeline status and to pipe i/o through the modules.

1. All pipeline modules must have the same output directory.
2. Only one pipeline can be run per output directory.
3. Each module run by the pipeline must have a unique job name (not specifying
   a name in the configs is preferred, and will use the directory name plus a
   suffix for each module).

Failed Jobs
-----------

The pipeline manager will keep a status of jobs that are submitted, running,
successful, or failed. If any jobs fail in a pipeline step, the pipeline will
wait until all jobs in that step have completed, then raise a failed message.
Error messages can be found in the stdout/stderr files belonging to the
respective failed job(s). The user can re-submit the full pipeline job and
only the jobs that failed will be re-run. If full modules had previously
finished successfully, those modules will be skipped.

File Inputs
-----------

There are several files beyond the NSRDB resource data used in this example
that are too big to be stored on github:

1. ``conus_trans_lines_cache_064_sj_infsink.csv`` in
   ``config_supply-curve.json`` is a transmission feature table from the reV
   database.
2. ``rev_conus_exclusions.h5`` in ``config_aggregation.json`` is an h5
   exclusions file containing exclusion layers for CONUS.
