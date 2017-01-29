#!/bin/bash

# Select the GPU that we've been allocated
export THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1'
export CUDA_ROOT=$CUDA_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64 

#! application name
application="python Run.py"

#! Run options for the application
options=""

###############################################################
### You should not have to change anything below this line ####
###############################################################

#! change the working directory (default is home directory)


$application $options
