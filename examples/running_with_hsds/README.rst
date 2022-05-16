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
    hs_api_key = 3K3JQbjZmWctY0xmIfSYvYgtIcM3CN0cb1Y2w9bf

*The example API key here is for demonstration and is rate-limited per IP. To
get your own API key, visit https://developer.nrel.gov/signup/*

*Please note that our HSDS service is for demonstration purposes only, if you
would like to use HSDS for production runs of reV please setup your own
service: https://github.com/HDFGroup/hsds and point it to our public HSDS
bucket: s3://nrel-pds-hsds*

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
    gen = Gen.reV_run('windpower', pp, sam_file, res_file, max_workers=1,
                      out_fpath=None, output_request=('cf_mean', 'cf_profile'))
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
    gen = Gen.reV_run('pvwattsv5', pp, sam_file, res_file,
                      max_workers=1, out_fpath=None,
                      output_request=('cf_mean', 'cf_profile'))
    print(gen.out['cf_mean'])

    [0.183 0.166 0.177 0.175 0.167 0.183 0.176 0.175 0.176 0.177]

Command Line Interface (CLI)
----------------------------

`reV-gen <https://nrel.github.io/reV/_cli/reV-gen.html#rev-gen>`_
can also be run from the command line and will output the results to an .h5
file that can be read with `rex.resource.Resource <https://nrel.github.io/rex/rex/rex.resource.html#rex.resource.Resource>`_.

windpower
+++++++++

Compute wind capacity factors for a given set of latitude and longitude
coordinates:

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=reV/tests/data
    res_file=/nrel/wtk/conus/wtk_conus_2012.h5
    sam_file=${TESTDATADIR}/SAM/wind_gen_standard_losses_0.json

    reV-gen direct --tech=windpower --res_file=${res_file} --sam_files=${sam_file} --lat_lon_coords 41.77 -71.74 local

pvwatts
+++++++

NOTE: ``pvwattsv5`` and ``pvwattsv7`` are both available from reV.

Compute pvcapacity factors for all resource gids in Rhode Island:

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=../tests/data
    res_file=/nrel/nsrdb/v3/nsrdb_2012.h5
    sam_file=${TESTDATADIR}/SAM/naris_pv_1axis_inv13.json

    reV-gen direct --tech=pvwattsv5 --res_file=${res_file} --sam_files=${sam_file} --region="Rhode Island" --region_col=state local

Setting up an HSDS Local Server on AWS EC2
------------------------------------------

You can stand up a local HSDS server on an EC2 instance to improve the HSDS throughput versus the NREL developer API. Generally you should follow `these instructions <https://github.com/HDFGroup/hsds/blob/master/docs/docker_install_aws.md>`_ from the HSDS documentation. Here are a few additional tips and tricks to get everything connected to the NREL bucket:

If you need to install docker and docker-compose on your EC2 instance (if not already installed). You can run ``docker run hello-world`` to test your docker install.

.. code-block:: bash

    sudo amazon-linux-extras install -y docker
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    sudo service docker start

Your ``~/.hscfg`` file should look like this (feel free to change the ``hs_username`` and ``hs_password``):

.. code-block:: bash

    # local hsds server
    hs_endpoint = http://localhost:5101
    hs_username = admin
    hs_password = admin
    hs_api_key = None
    hs_bucket = nrel-pds-hsds


The following environment variables must be set:

.. code-block:: bash

    export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
    export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    export BUCKET_NAME=${YOUR_S3_BUCKET_NAME_HERE}
    export AWS_REGION=us-west-2
    export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com/
    export HSDS_ENDPOINT=http://localhost:5101
    export LOG_LEVEL=INFO

A few miscellaneous tips:

#. You can list the available docker images with ``docker images``
#. You can delete the docker HSDS image with ``docker rmi $IMAGE_ID`` (useful to reset the docker image)
#. If you have AWS permissions issues try using a non-root IAM user with the corresponding AWS credentials as environment variables
#. You can stand up parallel docker HSDS servers on your EC2 instance by running ``sh runall.sh -8``
#. You can also set ``hs_endpoint = local`` in your ``~/.hscfg`` file to have h5pyd automatically spin up a local HSDS server as it opens a file handler. You still need to set all of the other environment variables for this to work.

Other Resources
---------------

For more HSDS examples please see: https://github.com/NREL/hsds-examples
