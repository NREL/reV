Using HSDS
==========

The Highly Scalable Distributed Service (HSDS) is a cloud optimized API to
enable access to .h5 files hosted on AWS. The HSDS software was developed by
the `HDF Group <https://www.hdfgroup.org/>`_ and is hosted on Amazon Web
Services (AWS) using a combination of EC2 (Elastic Compute) and S3 (Scalable
Storage Service). You can read more about the HSDS service
`in this slide deck <https://www.slideshare.net/HDFEOS/hdf-cloud-services>`_.
You can use the NREL developer API as the HSDS endpoint for small workloads
or stand up your own HSDS local server (instructions further below) for an
enhanced parallelized data experience.

You might also be interested in these examples of how to set up your own `local HSDS server <https://nrel.github.io/rex/misc/examples.hsds.html#setting-up-a-local-hsds-server>`_ and how to run reV on an `AWS parallel cluster <https://nrel.github.io/reV/misc/examples.aws_pcluster.html>`_.

Setting up HSDS
---------------

To get started install the h5pyd library:

.. code-block:: bash

    pip install h5pyd

Next, configure h5pyd by running ``hsconfigure`` from the command line, or by
creating a configuration file at ``~/.hscfg``:

.. code-block:: bash

    hsconfigure
    hs_endpoint = https://developer.nrel.gov/api/hsds
    hs_username =
    hs_password =
    hs_api_key = {YOUR_API_KEY_HERE}

To get your own API key, visit https://developer.nrel.gov/signup/

Please note that our HSDS service is for demonstration purposes only. The API in the example above is hosted on an NREL server and will have limits on the amount of data you can access via HSDS. It is common to get an error: ``OSError: Error retrieving data: None errors`` if you attempt to access too much data or if the server is busy. Here are two references for scaling reV using HSDS and AWS:

#. `Setup your own HSDS server on your personal computer <https://nrel.github.io/rex/misc/examples.hsds.html#>`_
#. `Run reV on the AWS Parallel Cluster Infrastructure <https://nrel.github.io/reV/misc/examples.aws_pcluster.html>`_

Using HSDS with reV
-------------------

Once h5pyd has been installed and configured, `rex <https://github.com/nrel/rex>`_
can pull data directly from AWS using `HSDS <https://github.com/NREL/hsds-examples>`_
To access the resource data used by reV (NSRDB or WTK) you have to turn on the
``hsds`` flag in the `resource handlers <https://nrel.github.io/rex/rex/rex.renewable_resource.html>`_:

.. code-block:: python

    nsrdb_file = '/nrel/nsrdb/v3/nsrdb_2013.h5'
    with rex.Resource(nsrdb_file, hsds=True) as f:
        meta_data = f.meta
        time_index = f.time_index

reV Gen
-------

reV generation (`reV.Gen <https://nrel.github.io/reV/_autosummary/reV.generation.html>`_)
will automatically infer if a file path is locally on disk or from HSDS.

Note that for all of these examples, the ``sam_file`` input points to files in
the
`reV test directory <https://github.com/NREL/reV/tree/master/tests/data/SAM>`_
that may not be copied in your install. You may want to download the relevant
SAM system configs from that directory and point the ``sam_file`` variable to
the correct filepath on your computer.

windpower
+++++++++

Compute wind capacity factors for a given set of latitude and longitude
coordinates:

.. code-block:: python

    import os
    import numpy as np
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen
    from rex import init_logger

    init_logger('reV', log_level='DEBUG')

    lat_lons = np.array([[ 41.25, -71.66],
                         [ 41.05, -71.74],
                         [ 41.45, -71.66],
                         [ 41.97, -71.78],
                         [ 41.65, -71.74],
                         [ 41.53, -71.7 ],
                         [ 41.25, -71.7 ],
                         [ 41.05, -71.78],
                         [ 42.01, -71.74],
                         [ 41.45, -71.78]])

    res_file = '/nrel/wtk/conus/wtk_conus_2012.h5'  # HSDS 'file' path
    sam_file = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_file)
    gen = Gen('windpower', pp, sam_file, res_file,
              output_request=('cf_mean', 'cf_profile'))
    gen.run(max_workers=1)
    print(gen.out['cf_profile'])

    [[0.319 0.538 0.287 ... 0.496 0.579 0.486]
     [0.382 0.75  0.474 ... 0.595 0.339 0.601]
     [0.696 0.814 0.724 ... 0.66  0.466 0.677]
     ...
     [0.833 0.833 0.823 ... 0.833 0.833 0.833]
     [0.782 0.833 0.833 ... 0.833 0.833 0.833]
     [0.756 0.801 0.833 ... 0.833 0.833 0.833]]

pvwatts
+++++++

NOTE: ``pvwattsv5`` and ``pvwattsv7`` are both available from reV.

Compute pvcapacity factors for all resource gids in a Rhode Island:

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen
    from rex import init_logger

    init_logger('reV', log_level='DEBUG')

    regions = {'Rhode Island': 'state'}

    res_file = '/nrel/nsrdb/v3/nsrdb_2012.h5'  # HSDS 'file' path
    sam_file = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')

    pp = ProjectPoints.regions(regions, res_file, sam_file)
    gen = Gen('pvwattsv5', pp, sam_file, res_file,
              output_request=('cf_mean', 'cf_profile'))
    gen.run(max_workers=1)
    print(gen.out['cf_mean'])

    [0.183 0.166 0.177 0.175 0.167 0.183 0.176 0.175 0.176 0.177]

Command Line Interface (CLI)
----------------------------

`reV-gen <https://nrel.github.io/reV/_cli/reV-gen.html#rev-gen>`_
can also be run from the command line and will output the results to an .h5
file that can be read with `rex.resource.Resource <https://nrel.github.io/rex/rex/rex.resource.html#rex.resource.Resource>`_.