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

Models that are implemented in EEGain for now :-

- [EEGNet](https://arxiv.org/abs/1611.08024)
- [TSception](https://ieeexplore.ieee.org/document/9762054)
- [DeepConvNet](https://www.researchgate.net/publication/318965745_Deep_learning_with_convolutional_neural_networks_for_EEG_decoding_and_visualization_Convolutional_Neural_Networks_in_EEG_Analysis)
- [ShallowConvNet](https://www.researchgate.net/publication/318965745_Deep_learning_with_convolutional_neural_networks_for_EEG_decoding_and_visualization_Convolutional_Neural_Networks_in_EEG_Analysis)

Datasets that are implemented in EEGain for now :-

- [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- [MAHNOB](https://mahnob-db.eu/)
- [AMIGOS](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/)
- [DREAMER](https://ieeexplore.ieee.org/document/7887697)
- [Seed](https://bcmi.sjtu.edu.cn/home/seed/seed.html)
- [Seed-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)

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

<table>
  <tr>
    <th style="width: 200px;">Argument</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><pre>--data_path</pre></td>
    <td>Specifies the directory where the data files are saved. You can check exact path for each dataset below in "Key Arguments to Modify" section.</td>
  </tr>
  <tr>
    <td><pre>--data_config</pre></td>
    <td>Specifies the dataset config that you want to load with the default arguements present in the <code>config.py</code> file.</td>
  </tr>
  <tr>
    <td><pre>--log_dir</pre></td>
    <td>Specifies the directory where the log files will be saved.</td>
  </tr>
  <tr>
    <td><pre>--overal_log_file</pre></td>
    <td>Specifies the name of the log file that will be created.</td>
  </tr>
  <tr>
    <td><pre>--label_type</pre></td>
    <td>
      Specifies whether data is separated into classes based on valence or arousal. This argument has no effect on the Seed and Seed IV dataset because these datasets have fixed splits based on categorical labels. You can choose the following options:<br><br>
      - <code>V</code>: Valence<br>
      - <code>A</code>: Arousal
    </td>
  </tr>
  <tr>
    <td><pre>--num_epochs</pre></td>
    <td>Sets the number of epochs for training.</td>
  </tr>
  <tr>
    <td><pre>--batch_size</pre></td>
    <td>Specifies the batch size for training.</td>
  </tr>
  <tr>
    <td><pre>--lr</pre></td>
    <td>Specifies the learning rate for training.</td>
  </tr>
  <tr>
    <td><pre>--sampling_r</pre></td>
    <td>Specifies the sampling rate of the EEG data.</td>
  </tr>
  <tr>
    <td><pre>--window</pre></td>
    <td>Specifies the length of the EEG segments (in seconds).</td>
  </tr>
  <tr>
    <td><pre>--overlap</pre></td>
    <td>Specifies the overlap between the EEG segments (in seconds).</td>
  </tr>
  <tr>
    <td><pre>--weight_decay</pre></td>
    <td>Specifies the weight decay ratio for regularization.</td>
  </tr>
  <tr>
    <td><pre>--label_smoothing</pre></td>
    <td>Smoothing factor applied to the labels to make them less extreme.</td>
  </tr>
  <tr>
    <td><pre>--dropout_rate</pre></td>
    <td>Probability at which outputs of the layer are dropped out.</td>
  </tr>
  <tr>
    <td><pre>--num_classes</pre></td>
    <td>Specifies the number of classes of the classification problem. Set this argument 2 for AMIGOS, MAHNOB, DEAP and DREAMER; 3 for SEED and 4 for SEED IV.</td>
  </tr>
  <tr>
    <td><pre>--channels</pre></td>
    <td>Specifies the number of channels for the dataset. Set this argument to 14 for AMIGOS and DREAMER, 32 for MAHNOB and DEAP, and to 62 for SEED and SEED IV.</td>
  </tr>
  <tr>
    <td><pre>--split_type</pre></td>
    <td>
      Specifies the type of train-test splitting. There are three different types of splitting:<br><br>
      - <code>LOTO</code>: Leave one trial out. Use this split for the person-dependent task.<br>
      - <code>LOSO</code>: Leave one subject out. Use this split for the person-independent task.<br>
      - <code>LOSO_Fixed</code>: Creates a fixed 75/25 train-test split that is mandatory for the person-independent task.
    </td>
  </tr>
  <tr>
    <td><pre>--train_val_split</pre></td>
    <td>Specifies the training and validation split for the data (default value = 0.8).</td>
  </tr>
  <tr>
    <td><pre>--random_seed</pre></td>
    <td>Sets the random seed value to ensure reproducibility (default value = 2025).</td>
  </tr>
  <tr>
    <td><pre>--log_predictions</pre></td>
    <td>Specifies whether the user wants to log the predictions from the chosen model and dataset combination for the Test sets. Set this argument to <code>True</code> if you want to log the predicitions, otherwise leave it out or manually set to <code>False</code>.</td>
  </tr>
  <tr>
    <td><pre>--log_predictions_dir</pre></td>
    <td>Specifies the directory where the logged predicitions will be stored in CSV format.</td>
  </tr>
</table>


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