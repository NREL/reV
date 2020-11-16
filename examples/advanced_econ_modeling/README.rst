reV Spatial Economics with SAM Single Owner Model
=================================================

This example set shows how several of the reV features (batching, pipeline,
site-data) can be used in concert to create complex spatially-variant economic
analyses.

This example modifies the tax rate and PPA price inputs for each state.
More complex input sets on a site-by-site basis can be easily generated using a
similar site_data input method.

Workflow Description
--------------------

The batching config in this example represents the high-level executed module.
The user executes the following command:

.. code-block:: bash

    reV -c "../config_batch.json" batch

This creates and executes three batch job pipelines. You should be able to see
in ``config_batch.json`` how the actual input generation files are
parameterized. This is the power of the batch module - it's sufficiently
generic to modify ANY key-value pairs in any ``.json`` file, including other
config files.

The first module executed in each job pipeline is the econ module. This example
shows how the site-specific input ``.csv`` can be used (see the "site_data" key
in the ``config_econ.json`` file).

The ``site_data.csv`` file sets site-specific input data corresponding to the
gids in the project points file. Data inputs keyed by each column header in the
``site_data.csv`` file will be added to or replace an input in the
"tech_configs" ``.json`` files (sam_files).
