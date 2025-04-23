#!/bin/bash
# !python run_client.py --model_name="TSception"
## example model names : EEGNet, TSception, DeepConvNet, ShallowConvNet, RANDOM_most_occurring, RANDOM_class_distribution
## data configs : DEAPConfig, MAHNOBConfig, SeedIVConfig, AMIGOSConfig, DREAMERConfig, SeedConfig
## example data names: DEAP, MAHNOB, SeedIV, AMIGOS, DREAMER, Seed
## set channel to 14 for amigos and dreamer and 32 for mahnob and deap, 62 for seedIV and seed 
python run_cli.py \
--model_name=ShallowConvNet \
--data_name=MAHNOB \
--data_path='path_to_data' \
--data_config=MAHNOBConfig \
--split_type="LOTO" \
--num_classes=2 \
--sampling_r=128 \
--channels=32 \
--window=4 \
--overlap=0 \
--label_type="V" \
--num_epochs=200 \
--batch_size=32 \
--lr=0.001 \
--weight_decay=0 \
--label_smoothing=0.01 \
--dropout_rate=0.5 \
--train_val_split=0.8 \
--log_dir="logs/" \
--overal_log_file="log_file_name.txt" \
--log_predictions=True \
--log_predictions_dir="path_to_store_predictions" \