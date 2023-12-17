#!/bin/bash

# works only for one dataset. change dataset, class_names, data class

python run_cli.py \
--model_name=TSception \
--data_path='Sessions' \
--log_dir=logs/ \
--label_type=V \
--num_epochs=2 \
--batch_size=32 \
--lr=0.0001 \
--s_rate=128 \
--window=4 \
--weight_decay=0.01 \
--label_smoothing=0.01 \
--dropout_rate=0.5
--num_classes=2 \
--sampling_r=128 \
--num_t=15 \
--num_s=15 \
--hidden=32 \


python run_cli.py \
--model_name=TSception \
--data_path='Sessions' \
--log_dir=logs_lr/ \
--label_type=V \
--num_epochs=2 \
--batch_size=32 \
--lr=0.001 \
--s_rate=128 \
--window=4 \
--weight_decay=0.01 \
--label_smoothing=0.01 \
--dropout_rate=0.5
--num_classes=2 \
--sampling_r=128 \
--num_t=15 \
--num_s=15 \
--hidden=32 \
