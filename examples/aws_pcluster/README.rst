Running reV on AWS Parallel Cluster HPC Infrastructure
======================================================

reV was originally designed to run on the NREL high performance computer (HPC), but you can now run reV on AWS using the NREL renewable energy resource data (the NSRDB and WTK) on S3. This example will guide you through how to set up reV on an AWS HPC environment with dynamically scaled EC2 compute resources and input resource data from HSDS.

If you plan on only running reV for a handful of sites (less than 100), first
check out our `running with HSDS example
<https://github.com/NREL/reV/tree/main/examples/running_with_hsds>`_,
which will be a lot easier to get started with.

Note that everything should be done in AWS region us-west-2 (Oregon) since this is where the NSRDB and WTK data live on S3.

Instructions
------------

1. Get started with the `AWS HPC Overview <https://www.hpcworkshops.com/01-hpc-overview.html>`_.
2. Set up a `Cloud9 IDE <https://www.hpcworkshops.com/02-aws-getting-started.html>`_.
3. Set up an `AWS Parallel Cluster <https://www.hpcworkshops.com/03-hpc-aws-parallelcluster-workshop.html>`_.

    a. Use the `rev-pcluster-config.ini <https://github.com/NREL/reV/blob/gb/aws/examples/aws_pcluster/rev-pcluster-config.ini>`_ file as an example.
    b. Choose a basic instance for head node (``master_instance_type``), e.g. t2.micro, t2.large, c5.large, or c5.xlarge. Note that a t2 instance is free-tier eligible and is probably sufficient for the pcluster login node which will not be doing any of the compute or storage.
    c. Choose a shared EBS storage volume (this is the ``/shared/`` volume) with a "gp2" (``volume_type``) which can have SSD storage ranging from 1GB-16TB (``volume_size``).

4. Optional, set up an `HPC parallel filesystem <https://www.hpcworkshops.com/04-amazon-fsx-for-lustre.html>`_.

    a. Seems like EBS is probably fine and FSx is unnecessary for non-IO-intensive reV modules. Generation will source resource data from HSDS and so is probably fine with EBS. SC-aggregation is probably fine with EBS if you set ``pre_extract_inclusions=True``.

5. `Login to your cluster <https://www.hpcworkshops.com/03-hpc-aws-parallelcluster-workshop/07-logon-pc.html>`_.
6. Get `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ and install.

    a. ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
    b. ``sh Miniconda3-latest-Linux-x86_64.sh``

7. Install reV (``pip install reV`` or ``pip install -e .`` from cloned repo: https://github.com/NREL/reV)

    a. You need to clone the reV repo to get the ``aws_pcluster`` example described below. reV example files do not ship with the pypi package.
    b. If you are cloning the repo you will have to first add the pcluster public ssh key (``cat ~/.ssh/id_rsa.pub``) to your github ssh keys.
    c. You should probably put the reV repo in the ``/shared/`` volume so that the ``aws_pcluster`` example project directory is in the large EBS storage volume shared between compute nodes.

8. Configure HSDS by running ``hsconfigure`` from the command line if you plan to source NSRDB or WTK data from S3 (`more details here <https://github.com/NREL/reV/tree/main/examples/running_with_hsds>`_). You will also want to configure an HSDS lambda function if you want to run anything beyond just a handful of reV sites (see HSDS lambda instructions below).
9. Try running the reV ``aws_pcluster`` example:

    a. ``cd /shared/reV/examples/aws_pcluster``
    b. Either of the following:
        i. ``reV -c config_pipeline.json pipeline``
        ii. ``reV -c config_pipeline.json pipeline --monitor``
    c. Check the slurm queue with ``squeue`` and the compute cluster status in the ec2 console or with ``sinfo``
    d. Jobs will be in state "CF" (configuring) while the nodes spin up (this can take several minutes)

10. Note that the NREL HSDS development server has a very limited throughput
    and you may see 503 errors if too many requests are being processed. The
    solution is to set up an HSDS lambda function (see instructions below).


Setting up an HSDS Lambda Service
---------------------------------

These instructions are generally copied from the `HSDS Lambda README
<https://github.com/HDFGroup/hsds/blob/master/docs/aws_lambda_setup.md>`_ with
a few modification after trying to actually run through the setup. It seems you
cannot currently use the public ECR container image from the HSDS ECR repo so
the first few bullets are instructions on how to set up your own HSDS image and
push to a private ECR repo.

1. Clone the `HSDS repository <https://github.com/HDFGroup/hsds>`_ into your Cloud9 environment.
2. You may need to `resize your EBS volume
   <https://docs.aws.amazon.com/cloud9/latest/user-guide/move-environment.html#move-environment-resize>`_.
3. In the AWS Management Console, create a new ECR repository called
   "hslambda". Keep the default private repo settings.
4. Create an HSDS image and push to your hslambda ECR repo. This sublist is a
   combination of commands from the ECR push commands and the HSDS build
   instructions (make sure you use the actual push commands from your ECR repo 
   with the actual region, repository name, and aws account id):

    a. ``aws ecr get-login-password --region region | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com``
    b. ``sh lambda_build.sh``
    c. ``docker tag hslambda:latest aws_account_id.dkr.ecr.region.amazonaws.com/my-repository:tag``
    d. ``docker push aws_account_id.dkr.ecr.region.amazonaws.com/my-repository:tag``

5. You should now see your new image appear in your hslambda ECR repo in the
   AWS Console. Get the URI from this image.
6. In the AWS Management Console, select the Lambda service for your desired region (us-west-2 Oregon).
7. Click "Create Function".
8. Choose the "Container Image" option.
9. Enter a function name (e.g. "hslambda")
10. Select "Browse Image" and find the image you just uploaded, or paste in the corresponding URI.
11. Select "Create Function" and wait for the image to load.
12. You should see a banner saying you've successfully created the "hslambda" function. Yay!
13. Change the configuration to use at least 1024MB of memory and a timeout of at least 30 seconds.
14. Also in the "Configuration" tab, select "Environment variables", click the
    "Edit" button, and then the "Add environment variable" button. Enter a key
    of "AWS_S3_GATEWAY" and a value corresponding to the S3 endpoint for your
    region. E.g. "http://s3.us-west-2.amazonaws.com" for us-west-2
15. Select "Permissions" -> "Execution Role" -> "Edit" and create a new role that includes S3 read only access.
16. Select the "Test" tab and click on the "Test" button. You should see a successful run with a ``status_code`` of 200 and an output like this:

.. code-block::

    {
      "isBase64Encoded": false,
      "statusCode": 200,
      "headers": "{\"Content-Type\": \"application/json; charset=utf-8\", \"Content-Length\": \"323\", \"Date\": \"Tue, 23 Nov 2021 22:27:08 GMT\", \"Server\": \"Python/3.8 aiohttp/3.8.1\"}",
      "body": "{\"start_time\": 1637706428, \"state\": \"READY\", \"hsds_version\": \"0.7.0beta\", \"name\": \"HSDS on AWS Lambda\", \"greeting\": \"Welcome to HSDS!\", \"about\": \"HSDS is a webservice for HDF data\", \"node_count\": 1, \"dn_urls\": [\"http+unix://%2Ftmp%2Fhs1a1c917f%2Fdn_1.sock\"], \"dn_ids\": [\"dn-001\"], \"username\": \"anonymous\", \"isadmin\": false}"
    }

17. Add a API Gateway trigger, make sure it's a REST API with an API key
18. Enable `CloudWatch logging <https://aws.amazon.com/premiumsupport/knowledge-center/api-gateway-cloudwatch-logs/>`_ (optional but useful)
19. Set up a API usage plan including method throttling for your API key.
