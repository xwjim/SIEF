#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

model_name=HeterGSAN_GloVe
batch_size=32
test_batch_size=2
input_theta=${2--1}
dataset=dev

python3 test.py \
  --dataset docred \
  --use_model bilstm \
  --model_name ${model_name} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --word_emb_size 100 \
  --pre_train_word \
  --gcn_dim 256 \
  --gcn_layers 2 \
  --lstm_hidden_size 128 \
  --use_entity_type \
  --use_entity_id \
  --finetune_word \
  --graph_type mhgat \
  --activation relu \
  --use_dis_embed \
  --use_graph \
  --test_type ${dataset} \
  --pretrain_model checkpoint/HeterGSAN_Glove_base_best.pt \
  --input_theta ${input_theta}

