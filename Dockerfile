# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /reV
RUN mkdir -p /reV

# Copy package
COPY . /reV

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["reV"]
