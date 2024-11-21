#!/bin/bash
exec > install.log 2>&1

# Log environment variables
echo "Environment variables:" >> install.log
env >> install.log

# Log current working directory
echo "Current working directory:" >> install.log
pwd >> install.log

# upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt