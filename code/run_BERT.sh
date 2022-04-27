#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

model_name=HeterGSAN_BERT_base
lr=1e-3
bert_lr=1e-5
batch_size=5
test_batch_size=5
epoch=300
test_epoch=1
log_step=20
save_model_freq=100
negativa_alpha=5
clip=-1
weight_decay=0.0001
warmup_epoch=0

nohup python3 -u train.py \
  --dataset docred \
  --use_model bert \
  --model_name ${model_name} \
  --pretrain_model checkpoint/HeterGSAN_BERT_base_best.pt \
  --lr ${lr} \
  --bert_lr ${bert_lr} \
  --batch_size ${batch_size} \
  --clip ${clip} \
  --test_batch_size ${test_batch_size} \
  --epoch ${epoch} \
  --test_epoch ${test_epoch} \
  --log_step ${log_step} \
  --save_model_freq ${save_model_freq} \
  --negativa_alpha ${negativa_alpha} \
  --gcn_dim 808 \
  --gcn_layers 2 \
  --bert_hid_size 768 \
  --bert_path bert-base-uncased \
  --use_entity_type \
  --activation relu \
  --use_entity_id \
  --use_dis_embed \
  --graph_type mhgat \
  --use_wandb \
  --wandb_name g \
  --use_graph \
  --use_sief \
  --coslr \
  --weight_decay ${weight_decay} \
  --warmup_epoch ${warmup_epoch} \
  >logs/train_${model_name}.log 2>&1 &
