#!/bin/bash

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
sudo usermod -aG docker $USER
sudo service docker start
