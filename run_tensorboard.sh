#!/bin/bash
export TF_LOG_DIR="tf/log/"
​
# run a simple program that generates logs for tensorboard
mkdir -p ~/tf/log
​
# cluster_utils.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source cluster_utils.sh
start_cluster code_template_tensorboard.py cluster2
​
# defined in cluster_utils.sh to terminate the cluster
# terminate_cluster
​
tensorboard --logdir $TF_LOG_DIR
