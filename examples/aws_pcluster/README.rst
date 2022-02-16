Running reV on AWS Parallel Cluster HPC Infrastructure
======================================================

reV was originally designed to run on the NREL high performance computer (HPC), but you can now run reV on AWS using the NREL renewable energy resource data (the NSRDB and WTK) that lives on S3. This example will guide you through how to set up reV on an AWS HPC environment with dynamically scaled EC2 compute resources and input resource data sourced from S3 via HSDS.

If you plan on only running reV for a handful of sites (less than 100), first check out our `running with HSDS example <https://github.com/NREL/reV/tree/main/examples/running_with_hsds>`_, which will be a lot easier to get started with. Larger reV jobs require you stand up your own AWS parallel cluster and HSDS server. Very small jobs can be run locally using the NREL HSDS developer API.

Note that everything should be done in AWS region us-west-2 (Oregon) since this is where the NSRDB and WTK data live on S3.

Setting up an AWS Parallel Cluster
----------------------------------

#. Get started with the `AWS HPC Overview <https://www.hpcworkshops.com/01-hpc-overview.html>`_.
#. Set up a `Cloud9 IDE <https://www.hpcworkshops.com/02-aws-getting-started.html>`_.
#. Set up an `AWS Parallel Cluster <https://www.hpcworkshops.com/03-hpc-aws-parallelcluster-workshop.html>`_.

    #. Use the `rev-pcluster-config.ini <https://github.com/NREL/reV/blob/gb/aws/examples/aws_pcluster/rev-pcluster-config.ini>`_ file as an example.
    #. Choose a basic instance for head node (``master_instance_type``), e.g. t2.micro, t2.large, c5.large, or c5.xlarge. Note that a t2 instance is free-tier eligible and is probably sufficient for the pcluster login node which will not be doing any of the compute or storage.
    #. Choose a shared EBS storage volume (this is the ``/shared/`` volume) with a "gp2" (``volume_type``) which can have SSD storage ranging from 1GB-16TB (``volume_size``).

#. Optional, set up an `HPC parallel filesystem <https://www.hpcworkshops.com/04-amazon-fsx-for-lustre.html>`_.

    * Seems like EBS is probably fine and FSx is unnecessary for non-IO-intensive reV modules. Generation will source resource data from HSDS and so is probably fine with EBS. SC-aggregation is probably fine with EBS if you set ``pre_extract_inclusions=True``.

#. `Login to your cluster <https://www.hpcworkshops.com/03-hpc-aws-parallelcluster-workshop/07-logon-pc.html>`_ from your cloud9 IDE: ``pcluster ssh pcluster_name -i ~/.ssh/lab-3-key``
#. Get `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and install.

    #. ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
    #. ``sh Miniconda3-latest-Linux-x86_64.sh``

#. Set up an HSDS service. At this time, it is recommended that you use HSDS Local Servers on your compute cluster. See instructions below for details.
#. Install reV

    #. You need to clone the reV repo to get this ``aws_pcluster`` example. reV example files do not ship with the pypi package.
    #. You will have to first add the pcluster public ssh key (``cat ~/.ssh/id_rsa.pub``) to your github ssh keys.
    #. Put the reV repo in the ``/shared/`` volume so that the ``aws_pcluster`` project directory is in the large EBS storage volume shared between compute nodes.
    #. ``cd /shared/``
    #. ``git clone git@github.com:NREL/reV.git``
    #. ``cd /shared/reV/``
    #. ``pip install -e .``

#. Try running the reV ``aws_pcluster`` example:

    #. ``cd /shared/reV/examples/aws_pcluster``
    #. ``reV -c config_pipeline.json pipeline``
    #. Check the slurm queue with ``squeue`` and the compute cluster status in the ec2 console or with ``sinfo``
    #. Jobs will be in state ``CF`` (configuring) while the nodes spin up (this can take several minutes) and then ``R`` (running)

Notes on Running reV in the AWS Parallel Cluster
------------------------------------------------

#. If you don't configure a custom HSDS Service you will almost certainly see 503 errors from too many requests being processed. See the instructions below to configure an HSDS Service.
#. AWS EC2 instances usually have twice as many vCPUs as physical CPUs due to a default of two threads per physical CPU (at least for the c5 instances) (see ``disable_hyperthreading = false``). The pcluster framework treats each thread as a "node" that can accept one reV job. For this reason, it is recommended that you scale the ``"nodes"`` entry in the reV generation config file but keep ``"max_workers": 1``. For example, if you use two ``c5.2xlarge`` instances in your compute fleet, this is a total of 16 vCPUs, each of which can be thought of as a HPC "node" that can run one process at a time.
#. If you setup an HSDS local server but the parallel cluster ends up sending too many requests (some nodes but not all will see 503 errors), you can try upping the ``max_task_count`` in the ``~/hsds/admin/config/override.yml`` file.
#. If your HSDS local server nodes run out of memory (monitor with ``docker stats``), you can try upping the ``dn_ram`` or ``sn_ram`` options in the ``~/hsds/admin/config/override.yml`` file.
#. The best way to stop your pcluster is using ``pcluster stop pcluster_name`` from the cloud9 IDE (not ssh'd into the pcluster) and then stop the login node in the AWS Console EC2 interface (find the "master" node and stop the instance). This will keep the EBS data intact and not charge you for EC2 costs. When you're done with the pcluster you can call ``pcluster delete pcluster_name`` but this will also delete all of the EBS data.


Setting up HSDS Local Servers on your Compute Cluster
-----------------------------------------------------

The current recommended approach for setting up an HSDS service for reV is to start local HSDS servers on your AWS parallel cluster compute nodes. These instructions set up a shell script that each reV compute job will run on its respective compute node. The shell script checks that an HSDS local server is running, and will start one if not. These instructions are generally copied from the `HSDS AWS README <https://github.com/HDFGroup/hsds/blob/master/docs/docker_install_aws.md>`_ with a few modifications.

#. Make sure you have installed Miniconda but have not yet installed reV/rex.
#. Clone the `HSDS Repository <https://github.com/HDFGroup/hsds>`_. into your home directory in the pcluster login node: ``git clone git@github.com:HDFGroup/hsds.git`` (you may have to set up your ssh keys first).
#. Install HSDS by running ``python setup.py install`` from the hsds repository folder (running ``python setup.py install`` is currently required as the setup script does some extra magic over a pip installation).
#. Copy the password file: ``cp ~/hsds/admin/config/passwd.default ~/hsds/admin/config/passwd.txt`` and (optionally) modify any username/passwords you wish.
#. Create an HSDS config file at ``~/.hscfg`` with the following entries:

    .. code-block:: bash

        # Local HSDS server
        hs_endpoint = http://localhost:5101
        hs_username = admin
        hs_password = admin
        hs_api_key = None
        hs_bucket = nrel-pds-hsds

#. Copy the ``start_hsds.sh`` script from this example to your home directory in the pcluster login node.
#. Replace the following environment variables in ``start_hsds.sh`` with your values: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and ``BUCKET_NAME`` (note that you should use AWS keys from an IAM user with admin privileges and not your AWS console root user).
#. Optional, to test your HSDS local server config, do the following:

    #. Run the start script: ``sh ~/start_hsds.sh``
    #. Run ``docker ps`` and verify that there are 4 or more HSDS services active (hsds_rangeget_1, hsds_sn_1, hsds_head_1, and an hsds_dn_* node for every available core)
    #. Run ``hsinfo`` and verify that this doesn't throw an error
    #. Try running ``pip install h5pyd`` and then run the the h5pyd test (either the .py in this example or the h5pyd test snippet below).

#. Make sure this key-value pair is set in the ``execution_control`` block of the ``config_gen.json`` file: ``"sh_script": "sh ~/start_hsds.sh"``
#. Optional, copy the config override file: ``cp ~/hsds/admin/config/config.yml ~/hsds/admin/config/override.yml``, update any config lines in the ``override.yml`` file that you wish to change, and remove all other lines (see notes on ``max_task_count`` and ``dn_ram``).
#. You should be good to go! The line in the generation config file makes reV run the ``start_hsds.sh`` script before running the reV job. The script will install docker and make sure one HSDS server is running per EC2 instance.


Setting up an HSDS Kubernetes Service
-------------------------------------

Setting up your own HSDS Kubernetes service is one way to run a large reV job with full parallelization. This has not been trialed by the NREL team in full, but we have tested on the HSDS group's Kubernetes cluster. If you want to pursue this route, you can follow the HSDS repository instructions for `HSDS Kubernetes on AWS <https://github.com/HDFGroup/hsds/blob/master/docs/kubernetes_install_aws.md>`_.


Setting up an HSDS Lambda Service
---------------------------------

We've tested AWS Lambda functions as the HSDS service for reV workflows and we've found that Lambda functions require too much overhead to work well with the reV workflow. These instructions are included here for posterity, but HSDS-Lambda is _not_ recommended for the reV workflow.

These instructions are generally copied from the `HSDS Lambda README <https://github.com/HDFGroup/hsds/blob/master/docs/aws_lambda_setup.md>`_ with a few modifications.

It seems you cannot currently use the public ECR container image from the HSDS ECR repo so the first few bullets are instructions on how to set up your own HSDS image and push to a private ECR repo.

H5pyd cannot currently call a lambda function directly, so the instructions at the end show you how to set up an API gateway that interfaces between h5pyd and the lambda function.

Follow these instructions from your Cloud9 environment. None of this is directly related to the pcluster environment, except for the requirement to add the ``.hscfg`` file in the pcluster home directory.

#. Clone the `HSDS repository <https://github.com/HDFGroup/hsds>`_ into your Cloud9 environment.
#. You may need to `resize your EBS volume <https://docs.aws.amazon.com/cloud9/latest/user-guide/move-environment.html#move-environment-resize>`_.
#. In the AWS Management Console, create a new ECR repository called "hslambda". Keep the default private repo settings.
#. Create an HSDS image and push to your ``hslambda`` ECR repo. This sublist is a combination of commands from the ECR push commands and the HSDS build instructions (make sure you use the actual push commands from your ECR repo with the actual region, repository name, and aws account id):

    #. ``cd hsds``
    #. ``aws ecr get-login-password --region region | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com``
    #. ``sh lambda_build.sh``
    #. ``docker tag hslambda:latest aws_account_id.dkr.ecr.region.amazonaws.com/my-repository:tag``
    #. ``docker push aws_account_id.dkr.ecr.region.amazonaws.com/my-repository:tag``

#. You should now see your new image appear in your ``hslambda`` ECR repo in the AWS Console. Get the URI from this image.
#. In the AWS Management Console, go to the Lambda service interface in your desired region (us-west-2, Oregon).
#. Click "Create Function" -> Choose "Container Image" option, function name is ``hslambda``, use the Container Image URI from the image you just uploaded to your ECR repo, select "Create Function" and wait for the image to load.
#. You should see a banner saying you've successfully created the ``hslambda`` function. Yay!
#. Set the following in the configuration tab:

    #. Use at least 1024MB of memory (feel free to tune this later for your workload)
    #. Timeout of at least 30 seconds (feel free to tune this later for your workload)
    #. Use an execution role that includes S3 read only access
    #. Add an environment variable ``AWS_S3_GATEWAY``: ``http://s3.us-west-2.amazonaws.com``

#. Select the "Test" tab and click on the "Test" button. You should see a successful run with a ``status_code`` of 200 and an output like this:

    .. code-block::

        {
          "isBase64Encoded": false,
          "statusCode": 200,
          "headers": "{\"Content-Type\": \"application/json; charset=utf-8\", \"Content-Length\": \"323\", \"Date\": \"Tue, 23 Nov 2021 22:27:08 GMT\", \"Server\": \"Python/3.8 aiohttp/3.8.1\"}",
          "body": "{\"start_time\": 1637706428, \"state\": \"READY\", \"hsds_version\": \"0.7.0beta\", \"name\": \"HSDS on AWS Lambda\", \"greeting\": \"Welcome to HSDS!\", \"about\": \"HSDS is a webservice for HDF data\", \"node_count\": 1, \"dn_urls\": [\"http+unix://%2Ftmp%2Fhs1a1c917f%2Fdn_1.sock\"], \"dn_ids\": [\"dn-001\"], \"username\": \"anonymous\", \"isadmin\": false}"
        }

#. Now we need to create an API Gateway so that reV and h5pyd can interface with the lambda function. Go to the API Gateway page in the AWS console and do these things:

    #. Create API -> choose HTTP API (build)
    #. Add integration -> Lambda -> use ``us-west-2``, select your lambda function, use some generic name like ``hslambda-api``
    #. Configure routes -> Method is ``ANY``, the Resource path is ``$default``, the integration target is your lambda function
    #. Configure stages -> Stage name is ``$default`` and auto-deploy must be enabled
    #. Create and get the API's Invoke URL, something like ``https://XXXXXXX.execute-api.us-west-2.amazonaws.com``

#. Make a ``.hscfg`` file in the home dir (``/home/ec2-user/``) in your Cloud9 env. Make sure you also have this config in your pcluster filesystem. The config file should have these entries:

    .. code-block:: bash

        # HDFCloud configuration file
        hs_endpoint = https://XXXXXXX.execute-api.us-west-2.amazonaws.com
        hs_username = hslambda
        hs_password = lambda
        hs_api_key = None
        hs_bucket = nrel-pds-hsds

#. All done! You should now be able to run the ``aws_pcluster`` test sourcing data from ``/nrel/nsrdb/v3/nsrdb_{}.h5`` or the simple h5pyd test below.
#. Here are some summary notes for posterity:

    #. We now have a lambda function ``hslambda`` that will retrieve data from the NSRDB or WTK using the HSDS service.
    #. We have an API Gateway that we can use as an endpoint for API requests
    #. We have configured h5pyd with the ``.hscfg`` file to hit that API endpoint with the proper username, password, and bucket target
    #. reV will now retrieve data from the NSRDB or WTK in parallel requests to the ``hslambda`` function via h5pyd.
    #. Woohoo! We did it!

Simple H5PYD Test
-----------------

Here's a simple h5pyd test to make sure you can retrieve data from the NSRDB/WTK via HSDS. This python example should return a ``numpy.ndarray`` object with shape ``(17520,)``. Obviously you will need to install python and h5pyd before running this test.

.. code-block:: python

    from rex import init_logger
    import h5pyd
    import logging

    if __name__ == '__main__':
        logger = logging.getLogger(__name__)
        init_logger(__name__, log_level='DEBUG')
        fp = '/nrel/nsrdb/v3/nsrdb_2019.h5'
        with h5pyd.File(fp, logger=__name__) as f:
            data = f['ghi'][:, 0]
        print(data)
        print(type(data))
        print(data.shape)


Compute Cost Estimates
----------------------

Here are some initial compute cost results and estimates for running reV generation (the largest compute module in reV). All estimates are only for EC2 compute costs based on c5.2xlarge instances at the on-demand price of $0.34 per hour. These numbers are *rough* estimates! Consider making your own estimates before developing a budget. The EC2 costs could be reduced significantly if running in the EC2 spot market (see how to configure pcluster spot pricing `here <https://docs.aws.amazon.com/parallelcluster/latest/ug/compute-resource-section.html#compute-resource-spot-price>`_. The ``sites_per_worker`` input in the ``config_gen.json`` file will also influence the computational efficiency.

.. list-table:: reV PCluster Compute Costs (Empirical)
    :widths: auto
    :header-rows: 1

    * - Compute Module
      - Timesteps
      - Sites
      - Total Datum
      - Total Compute Time (hr)
      - Total EC2 Cost
      - Cost per Datum
    * - PVWattsv7
      - 35088
      - 1850
      - 6.49e7
      - 3.4
      - $1.15
      - 1.77e-8
    * - Windpower
      - 17544
      - 6268
      - 1.10e8
      - 1.2
      - $0.42
      - 3.79e-09

.. list-table:: CONUS Compute Costs (Estimated)
    :widths: auto
    :header-rows: 1

    * - Compute Module
      - Source Data
      - Timesteps (one year)
      - Sites
      - Total Datum
      - Total Compute Time (hr)
      - Total EC2 Cost
    * - PVWattsv7
      - NSRDB (4km, 30min)
      - 17520
      - ~5e05
      - 8.76e9
      - 457.12
      - $155.42
    * - Windpower
      - WTK (2km, 1hr)
      - 8760
      - ~2e6
      - 1.75e10
      - 195.21
      - $66.37
