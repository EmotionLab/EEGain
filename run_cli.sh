# !python run_client.py --model_name="TSception"
## example model names : EEGNet, TSception, DeepConvNet, Shallow
## example data names: DEAP, MAHNOB, SeedIV, AMIGOS, DREAMER
## set channel and n_chan to 14 for amigos and dreamer and 32 for mahnob and deap, seedIV - 62 
!python run_cli.py \
--model_name=ShallowConvNet \
--data_name=DREAMER \
--data_path='Desktop/eegain/EEGain/DREAMER.mat' \
--log_dir="logs/" \
--overal_log_file="logs_dreamer_loso.txt" \
--label_type="V" \
--num_epochs=200 \
--batch_size=16 \
--lr=0.001 \
--sampling_r=128 \
--window=4 \
--s_rate=128 \
--n_time=512 \
--weight_decay=0 \
--label_smoothing=0.01 \
--dropout_rate=0.5 \
--num_classes=2 \
--channels=14 \
--n_chan=14 \
--num_t=15 \
--num_s=15 \
--hidden=32 \
--ground_truth_threshold=4.5 \
--split_type="LOSO" \