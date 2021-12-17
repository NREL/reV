#!/bin/bash
# shellcheck disable=SC2155

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
export BUCKET_NAME=${YOUR_S3_BUCKET_NAME_HERE}
export AWS_REGION=us-west-2
export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com/
export HSDS_ENDPOINT=http://localhost:5101
export LOG_LEVEL=INFO
export EC2_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
export EC2_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)

echo On EC2 "$EC2_TYPE" with ID "$EC2_ID"

if [[ $EC2_TYPE = t2* ]]; then
    echo On EC2 login node "$EC2_TYPE", not starting HSDS service.

elif [[ ($EC2_TYPE = c*) && (-d ~/hsds) ]]; then
    echo On EC2 compute node "$EC2_TYPE", starting HSDS and Docker...
    # install docker if not found
    if ! command -v docker &> /dev/null; then
        sudo amazon-linux-extras install -y docker
    fi

    # install docker-compose if not found
    if ! command -v docker-compose &> /dev/null; then
        sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi

    # make sure docker is available
    sudo chmod 666 /var/run/docker.sock
    sudo groupadd docker
    sudo usermod -aG docker "$USER"
    sudo service docker start

    echo Starting HSDS local server...
    cd ~/hsds/ || exit
    sh runall.sh "$(nproc --all)"
    cd - || exit
fi
