#!/bin/bash

# --TEST: test only
# --cont: learning from trained model
# --wo_cross: without cross update among agants
# sbatch -w million4 --gres=gpu:1 --mail-type=FAIL --mail-user=keisuke198619@gmail.com ./run.sh -c 16 --mem=50G
# Yokohama
# GVRNN
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 16 --model GVRNN 
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 76 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 16 --model GVRNN
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 96 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 16 --model GVRNN
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 116 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 16 --model GVRNN
# VRNN
python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model MACRO_VRNN  
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 76 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model MACRO_VRNN 
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 96 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model MACRO_VRNN  
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 116 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model MACRO_VRNN 
# RNN
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model RNN_GAUSS  
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 76 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model RNN_GAUSS 
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 96 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model RNN_GAUSS  
# python -u main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 116 --batchsize 256 --n_epoch 10 -ev_th 50 --numProcess 16 --model RNN_GAUSS 


# python main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model GVRNN 
# python main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model MACRO_VRNN 
# python main.py --data jleague --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model RNN_GAUSS 

# competition
# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model GVRNN 
# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model MACRO_VRNN 
# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model RNN_GAUSS 

# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model GVRNN 
# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model MACRO_VRNN 
# python main.py --data jleague2 --n_GorS 2 --n_roles 3 --attack_defend -t_step 56 --batchsize 256 --n_epoch 20 -ev_th 50 --numProcess 32 --model RNN_GAUSS 
# python -u main.py --data soccer --n_GorS 7500 --attack_defend --n_roles 3 --batchsize 256 --n_epoch 20 -ev_th 70 --numProcess 16 --attack_defend --model MACRO_VRNN

# python -u main.py --data soccer --n_GorS 250 --hmm_iter 6 --attack_defend --n_roles 3 --batchsize 256 --n_epoch 20 -ev_th 70 --numProcess 16 --attack_defend --model MACRO_VRNN
# python -u main.py --data soccer --n_GorS 250 --hmm_iter 1 --n_roles 10 --batchsize 256 --n_epoch 10 -ev_th 70 --numProcess 16 --model MACRO_VRNN --wo_macro
# python -u main.py --data jleague --n_GorS 7500 --n_roles 10 --batchsize 256 --n_epoch 10 -ev_th 70 --numProcess 16 --model MACRO_VRNN --wo_macro
