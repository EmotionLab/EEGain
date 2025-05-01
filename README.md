<p align="center">
    <img src="book/images/eegain_logo.png" alt="Logo" width="500" />
    <br/>
    <br/>
    <a href="LICENSE"><img alt="CC BY 4.0 License" src="https://img.shields.io/badge/license-CC BY 4.0-blue.svg" /></a>
    <a><img alt="Latest Release" src="https://img.shields.io/badge/version-1.0.0-orange" /></a>
    <a href=""><img alt="Paper" src="https://img.shields.io/badge/Paper-Link-green" /></a>
</p>

---
<!-- Logos of collaborating institutes to be added -->
<p align="center">
    A joint venture by:-
    <br/><br/>
    <img src="book/images/dfki_logo.png" alt="DFKI" height="50">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="book/images/micm_logo.png" alt="MICM" height="50">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="book/images/inria_logo.png" alt="Inria" height="50">
</p>

---

## Run experiments for EEG-based emotion recognition for most popular datasets and models with just one command

### **1. Description**
EEG-based emotion recognition has become increasingly popular direction in recent years. Although lots of researchers are working on this task, running experiments is still very difficult. The main challenges are related to datasets and models. Running experiments on new datasets means that the researcher would need to implement it with pytorch or tensorflow, get detailed information how dataset is recorded and saved, and much more. Running experiments on new models is also tricky where the researcher would need to implement it from scratch with pytorch or tensorflow. This process, on one hand, takes too much time and can cause lots of bugs and effort. On the other hand it is not helpful for the researcher for furthering their research. To solve this problem, to make the process easier and get more researchers in this field, we created **EEGain**. It is a novel framework for EEG-based emotion recognition to run experiments on different datasets and models easily, with one command. You can implement your custom models and datasets too. 

Models that are implemented in EEGain for now - EEGNet, TSception, DeepConvNet and ShallowConvNet.

Datasets that are implemented in EEGain for now - DEAP, MAHNOB, AMIGOS, DREAMER, Seed and SeedIV.

Some other models and datasets are coming. 

### **2. QuickStart**
You can simply run the code in Google Colab. First you need to clone repo with this command:
```
!git clone https://github.com/EmotionLab/EEGain.git
```
Then you can run it with this command:
```
!python3 run_cli.py \
--model_name=ShallowConvNet \
--data_name=MAHNOB \
--data_path='...' \
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
--random_seed=2025 \
--log_dir="logs/" \
--overal_log_file="log_file_name.txt" \
--log_predictions=True \
--log_predictions_dir=".../..." \
```
Here you can change some important arguments. For example, to change dataset here you need to change just 5 arguments - data_name, data_path, data_config, num_classes and channels. 

You can see results on tensorboard using the logs stored in `logs/` directory.

### **3. How to run**
   1. clone the repo
   2. Enter in EEGain folder. run <code>pip install .</code>
   3. Change run_cli.sh based on dataset/splitting/model requirements
   4. Run run_cli.sh
   5. to check the results: <br>
      - run <code>tensorboard --logdir ./logs </code>
      - logs will be saved in **log.log** file as well<br>

You can adapt arguments within the sh file according to your specific intentions:

```--model_name``` - Selects a model. The implemented predefined models are:
- TSception
- EEGNet
- DeepconvNet
- ShallowConvNet
- RANDOM_most_occurring (for testing random baseline using most occuring class as the output)
- RANDOM_class_distribution (for testing random baseline using class distribution based output)

**NOTE:** The RANDOM_most_occurring model always predicts the most occurring class in the training set, so it is not recommended to use it for F1-score calculations. For F1-score calculations, please use the RANDOM_class_distribution model, that predicts a random class based on class distribution.

You can add your custom model as well.

| **Argument**               | **Description** |
|---------------------------|-----------------|
| `--data_path`             | Specifies the directory where the data files are saved. You can check the exact path for each dataset below in the "Key Arguments to Modify" section. |
| `--data_config`           | Specifies the dataset config that you want to load with the default arguments present in the `config.py` file. |
| `--log_dir`               | Specifies the directory where the log files will be saved. |
| `--overal_log_file`       | Specifies the name of the log file that will be created. |
| `--label_type`            | Specifies whether data is separated into classes based on valence or arousal. This has no effect on the Seed and Seed IV dataset. Options: `V` (Valence), `A` (Arousal). |
| `--num_epochs`            | Sets the number of epochs for training. |
| `--batch_size`            | Specifies the batch size for training. |
| `--lr`                    | Specifies the learning rate for training. |
| `--sampling_r`            | Specifies the sampling rate of the EEG data. |
| `--window`                | Specifies the length of the EEG segments (in seconds). |
| `--overlap`               | Specifies the overlap between the EEG segments (in seconds). |
| `--weight_decay`          | Specifies the weight decay ratio for regularization. |
| `--label_smoothing`       | Smoothing factor applied to the labels to make them less extreme. |
| `--dropout_rate`          | Probability at which outputs of the layer are dropped out. |
| `--num_classes`           | Specifies the number of classes. Use 2 for AMIGOS, MAHNOB, DEAP, DREAMER; 3 for SEED; 4 for SEED IV. |
| `--channels`              | Specifies the number of EEG channels. Use 14 for AMIGOS and DREAMER, 32 for MAHNOB and DEAP, and 62 for SEED and SEED IV. |
| `--split_type`            | Specifies the type of train-test splitting. Options: `LOTO` (Leave One Trial Out — for person-dependent tasks), `LOSO` (Leave One Subject Out — for person-independent tasks), and `LOSO_Fixed` (Fixed 75/25 train-test split, mandatory for person-independent tasks). |
| `--train_val_split`       | Specifies the training and validation split (default = 0.8). |
| `--random_seed`           | Sets the random seed value for reproducibility (default = 2025). |
| `--log_predictions`       | Whether to log the predictions for test sets. Set to `True` to enable logging. |
| `--log_predictions_dir`   | Specifies the directory where the logged predictions will be stored in CSV format. |



### **4. Key Arguments to Modify**

**SeedIV Setup:**
- Data Path: Ensure your directory structure follows "your_path_to/eeg_raw_data", containing three session folders. Each session folder must include .mat files. The "eeg_raw_data" folder should also contain "Channel Order.xlsx" and "ReadMe.txt" files.<br/>
- Data Name: SeedIV<br/>
- Channels: 62<br/>
- Number of Classes: 4

**Seed Setup:**
- Data Path: Use the structure "your_path_to/Preprocessed_EEG", which should contain .mat files, a channel-order.xlsx file, and a label.mat file.<br/>
- Data Name: Seed<br/>
- Channels: 62<br/>
- Number of Classes: 3

**MAHNOB Setup**: <br/>
- Data Path: Follow "your_path_to/mahnob-hci-raw/Sessions", with session-associated folders. Each session folder must have a .xml file for labels and a .bdf file.<br/>
- Data Name: MAHNOB <br/>
- Channels: 32 <br/>
- Number of Classes: 2
- For this dataset, `self.inception_window = [0.25, 0.125, 0.0625]` is automatically set for TSception model to replicate the TSception (2022) paper.

**DEAP Setup:** <br/>
- Data Path: Structure your directory as "your_path_to/data_preprocessed_python", which should contain .dat files. <br/>
- Data Name: DEAP <br/>
- Channels: 32 <br/>
- Number of Classes: 2

**DREAMER Setup:** <br/>
- Data Path: Ensure your file path is "your_path_to/DREAMER.mat". <br/>
- Data Name: DREAMER <br/>
- Channels: 14 <br/>
- Number of Classes: 2

**AMIGOS Setup:** <br/>
- Data Path: Organize your path as "your_path_to/AMIGOS/", which should lead to a Physiological recordings folder, then to a "Matlab Preprocessed Data" folder containing .mat files. <br/>
- Data Name: AMIGOS <br/>
- Channels: 14 <br/>
- Number of Classes: 2 <br/>


**[Struture of the framework](https://miro.com/app/board/uXjVMEB2nB0=/?share_link_id=710707650624)** 

### **5. Processing Time for Different Datasets on Various Models**:

When running different models (such as **DeepConvNet**, **ShallowConvNet**, **EEGnet**, and **Tsception**) on a subject-dependent scenario, the time it takes to process the data can vary a little, but it primarily depends on the specific dataset being used.

Our tests were conducted on Google Colab, utilizing a V100 GPU, with a batch size of 32 and a total of 30 epochs for each run. The following is an estimate of the time required to complete the entire pipeline, using a Leave-One-Trial-Out (LOTO) split, for each dataset:

**DREAMER** Dataset: Approximately 2 to 3 hours.

**DEAP** Dataset: About 2.5 to 3.5 hours.

**MAHNOB** Dataset: Roughly 2 to 3 hours.

**AMIGOS** Dataset: Around 5 to 6 hours.

estimated time for LOTO_Fixed approach is approximately 20 minutes for each dataset independently 

### **6. License**:
This code repository is licensed under the [CC BY 4.0 License](LICENSE).

### **7. Citation**: