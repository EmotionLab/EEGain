# !python run_client.py --model_name="TSception"
## example model names : EEGNet, TSception, DeepConvNet, ShallowConvNet, RANDOM
## example data names: DEAP, MAHNOB, SeedIV, AMIGOS, DREAMER
## set channel and n_chan to 14 for amigos and dreamer and 32 for mahnob and deap, seedIV - 62 
python run_cli.py \
--model_name=ShallowConvNet \
--data_name=MAHNOB \
--data_path='path_to_data' \
--split_type="LOTO" \
--num_classes=3 \
--sampling_r=128 \
--window=4 \
--label_type="V" \
--num_epochs=200 \
--batch_size=16 \
--lr=0.001 \
--weight_decay=0 \
--label_smoothing=0.01 \
--dropout_rate=0.5 \
--channels=32 \
--log_dir="logs/" \
--overal_log_file="log_file_name.txt" \
--log_predictions=True \
--log_predictions_dir="path_to_store_predictions" \