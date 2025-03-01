#!/bin/bash

# install requirements
echo "Installing requirements..."
conda activate mdt_env
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
pip install setuptools==57.5.0
cd pyhash-0.9.3
python setup.py build
python setup.py install
cd ..
pip install -r requirements.txt
echo "Requirements installed"