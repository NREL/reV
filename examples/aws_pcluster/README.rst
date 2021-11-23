Running reV on AWS Parallel Cluster HPC Infrastructure
======================================================

reV was originally designed to run on the NREL high performance computer (HPC).
Since the NREL renewable energy resource data (the NSRDB and WTK) are publicly
available on AWS S3 in their full volumes, you can now run reV on AWS. This
example will guide you through how to set up reV on an AWS HPC environment with
dynamically scaled EC2 compute resources and input resource data from HSDS.

If you plan on only running reV for a handful of sites (less than 100), first
check out our `Running with HSDS example
<https://github.com/NREL/reV/tree/main/examples/running_with_hsds>`_.

Instructions
------------

1. Get started with the `AWS HPC Overview <https://www.hpcworkshops.com/01-hpc-overview.html>`_.
2. Set up a `Cloud9 IDE <https://www.hpcworkshops.com/02-aws-getting-started.html>`_.
3. Set up an `AWS Parallel Cluster <https://www.hpcworkshops.com/03-hpc-aws-parallelcluster-workshop.html>`_.
    a. Choose a basic instance for head node, e.g. t2.micro, t2.large, c5.large, or c5.xlarge. Note that a t2 instance is free-tier eligible and is probably sufficient for the pcluster login node which will not be doing any of the compute or storage.
    b. Choose a shared EBS storage volume like a "gp2" which can have SSD storage ranging from 1GB-16TB (this is the ``/shared/`` volume)
4. Optional, set up an HPC parallel filesystem: https://www.hpcworkshops.com/04-amazon-fsx-for-lustre.html
    a. Seems like EBS is probably fine and FSx is unnecessary for non-IO-intensive reV modules. Generation will source resource data from HSDS and so is probably fine with EBS. SC-aggregation is probably fine with EBS if you set ``pre_extract_inclusions=True``.
5. Login to your cluster
6. Get miniconda and install.
    a. ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
    b. ``sh Miniconda3-latest-Linux-x86_64.sh``
7. Install reV (``pip install reV`` or ``pip install -e .`` from cloned repo: https://github.com/NREL/reV)
    a. Note that you need to clone the reV repo to get the ``aws_pcluster`` example described below. reV example files do not ship with the pypi package.
8. Configure HSDS if you plant to source NSRDB or WTK data from S3 (see note about HSDS below)
9. Try running the reV ``aws_pcluster`` example:
    a. ``cd ~/reV/examples/aws_pcluster``
    b. Either of the following:
        i. ``reV -c config_pipeline.json pipeline``
        ii. ``reV -c config_pipeline.json pipeline --monitor``
    c. Check the slurm queue with ``squeue`` and the compute cluster status in the ec2 console or with ``sinfo``
    d. Jobs will be in state "CF" (configuring) while the nodes spin up (this can take several minutes)
10. Note that the NREL HSDS development server has a very limited throughput and you may see 503 errors if too many requests are being processed. The solution is to set up your own HSDS server or configure HSDS lambda.


