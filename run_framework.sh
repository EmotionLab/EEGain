#!/bin/bash

# works only for one dataset. change dataset, class_names, data class

python run_cli.py --model_name=TSception --s_rate=128 --window=4 --data_path='/Users/raphaelkalandadze/Downloads/Sessions' \
--label_type=V --batch_size=32 --lr=0.0001 --weight_decay=0.01 --label_smoothing=0.01 \
--num_epochs=30 --log_dir=logs/ --num_classes=2 --sampling_r=128 --num_t=15 --num_s=15 --hidden=32 --dropout_rate=0.5

