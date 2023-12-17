#!/bin/bash

python run_cli.py --model_name=TSception --s_rate=128 --window=4 --data_path='/Users/raphaelkalandadze/Downloads/mahnob-hci-raw/Sessions' \
--label_type=V --class_names=low,high --batch_size=32 --lr=0.0001 --weight_decay=0.01 --label_smoothing=0.01 \
--num_epochs=30 --log_dir=logs/ --num_classes=2 --input_size=1,32,512 --sampling_r=128 --num_t=15 --num_s=15 --hidden=32 --dropout_rate=0.5