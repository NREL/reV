#!/bin/bash

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
export BUCKET_NAME=${YOUR_S3_BUCKET_NAME_HERE}
export AWS_REGION=us-west-2
export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com/
export HSDS_ENDPOINT=http://localhost:5101
export LOG_LEVEL=INFO
export EC2_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)

if [[ $EC2_TYPE = t2* ]]; then
    echo On EC2 login node $EC2_TYPE, not starting HSDS service.
elif [[ ($EC2_TYPE = c*) && (-d ~/hsds) ]]; then
    if [ -f ~/install_docker.sh ]; then
        sh ~/install_docker.sh
    fi
    echo On EC2 compute node $EC2_TYPE, starting HSDS service...
    cd ~/hsds/
    sh runall.sh "$(nproc --all)"
    cd -
fi
