# **EEGain**

## Run experiments for EEG-based emotion recognition for most popular datasets and models with just one command

### **Description**
EEG-based emotion recognition has become increasingly popular direction in recent years. Although lots of researchers are working on this task, running experiments is still very difficult. The main challenges are related to dataset and model. Running experiment on new dataset means that researcher should implement it with pytorch or tensorflow, get detailed information how dataset is recorded and saved and many more. Running experiment on new model is also tricky, researcher should implement it from scratch with pytorch or tensorflow. This process on one hand takes too much time and can cause lots of bugs and effort, on the other hand it is not helpful for researcher for further research. To solve this problem, make process easier and get more researchers in this field, we created EEGain, is a novel framework for EEG-based emotion recognition to run experiments on different datasets and models easily, with one command. You can implement your custom models and datasets too. 

Models that are implemented in EEGain for now - EEGNet, TSception, DeepConvNet and ShallowConvNet.

Datasets that are implemented in EEGain for now - DEAP, MAHNOB, AMIGOS, DREAMER, Seed and SeedIV.

Some other models and datasets are comming. 

### **QuickStart**
You can simply run the code in Google Colab. First you need to clone repo with this command:
```
!git clone https://github.com/EmotionLab/EEGain.git
```
Then you can run it with this command:
```
!python3 run_cli.py \
--model_name=TSception \
--data_name=MAHNOB \
--data_path="..." \
--log_dir="logs/" \
--overal_log_file="logs_mahnob_loto.txt" \
--label_type="V" \
--num_epochs=2 \
--batch_size=32 \
--lr=0.001 \
--sampling_r=128 \
--window=4 \
--weight_decay=0 \
--label_smoothing=0.01 \
--dropout_rate=0.5 \
--num_classes=2 \
--channels=32 \
--split_type="LOTO"
```
Here you can change some important arguments. For example, to change dataset here you need to change just 4 arguments - data_name, data_path, num_classes and channels. 
You can see results on tensorboard. 

### **How to run**
   1. clone the repo
   2. Enter in EEGain folder. run <code>pip install .</code>
   3. Change run_cli.sh based on dataset/splitting/model requirements
   4. Run run_cli.sh
   5. to check the results: <br>
      - run <code>tensorboard --logdir ./logs </code>
      - logs will be saved in **log.log** file as well<br>
   
### **Key Arguments to Modify**

**SeedIV Setup:**
- Data Path: Ensure your directory structure follows "your_path_to/eeg_raw_data", containing three session folders. Each session folder must include .mat files.<br/>
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

**DEAP Setup:** <br/>
- Data Path: Structure your directory as "your_path_to/data_preprocessed_python", which should contain .mat files. <br/>
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

### **Processing Time for Different Datasets on Various Models**:

When running different models (such as **DeepConvNet**, **ShallowConvNet**, **EEGnet**, and **Tsception**) on a subject-dependent scenario, the time it takes to process the data can vary a little, but it primarily depends on the specific dataset being used.

Our tests were conducted on Google Colab, utilizing a V100 GPU, with a batch size of 32 and a total of 30 epochs for each run. The following is an estimate of the time required to complete the entire pipeline, using a Leave-One-Trial-Out (LOTO) split, for each dataset:

**DREAMER** Dataset: Approximately 2 to 3 hours.

**DEAP** Dataset: About 2.5 to 3.5 hours.

**MAHNOB** Dataset: Roughly 2 to 3 hours.

**AMIGOS** Dataset: Around 5 to 6 hours.

estimated time for LOTO_Fixed approach is approximately 20 minutes for each dataset independently 
