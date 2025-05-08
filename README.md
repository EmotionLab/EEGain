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

- <a href="https://arxiv.org/abs/1611.08024" target="_blank">EEGNet</a>
- <a href="https://ieeexplore.ieee.org/document/9762054" target="_blank">TSception</a>
- <a href="https://www.researchgate.net/publication/318965745_Deep_learning_with_convolutional_neural_networks_for_EEG_decoding_and_visualization_Convolutional_Neural_Networks_in_EEG_Analysis" target="_blank">DeepConvNet</a>
- <a href="https://www.researchgate.net/publication/318965745_Deep_learning_with_convolutional_neural_networks_for_EEG_decoding_and_visualization_Convolutional_Neural_Networks_in_EEG_Analysis" target="_blank">ShallowConvNet</a>

Datasets that are implemented in EEGain for now :-

- <a href="https://www.eecs.qmul.ac.uk/mmv/datasets/deap/" target="_blank">DEAP</a>
- <a href="https://mahnob-db.eu/" target="_blank">MAHNOB</a>
- <a href="https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/" target="_blank">AMIGOS</a>
- <a href="https://ieeexplore.ieee.org/document/7887697" target="_blank">DREAMER</a>
- <a href="https://bcmi.sjtu.edu.cn/home/seed/seed.html" target="_blank">Seed</a>
- <a href="https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html" target="_blank">Seed-IV</a>

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

<table>
  <tr>
    <th style="width: 200px;">Argument</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><pre>--model_name</pre></td>
    <td>
      Selects a model. The implemented predefined models are:<br><br>
      - <code>TSception</code><br>
      - <code>EEGNet</code><br>
      - <code>DeepconvNet</code><br>
      - <code>ShallowConvNet</code><br>
      - <code>RANDOM_most_occurring</code> (for testing random baseline using most occuring class as the output)<br>
      - <code>RANDOM_class_distribution</code> (for testing random baseline using class distribution based output)<br><br>
      <b>NOTE</b>: The <code>RANDOM_most_occurring</code> model always predicts the most occurring class in the training set, so it is not recommended to use it for F1-score calculations. For F1-score calculations, please use the <code>RANDOM_class_distribution</code> model, that predicts a random class based on class distribution.<br>
      You can add your custom model as well.
    </td>
  </tr>
  <tr>
    <td><pre>--data_name</pre></td>
    <td>
      Chooses a custom name or a name of predefined datasets. The predefined datasets are:<br><br>
      - <code>DEAP</code><br>
      - <code>MAHNOB</code><br>
      - <code>AMIGOS</code><br>
      - <code>DREAMER</code><br>
      - <code>Seed</code><br>
      - <code>SeedIV</code><br><br>
      You can add your custom dataset as well.
    </td>
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

**AMIGOS Setup:** <br/>
- Data Path: Organize your path as "your_path_to/AMIGOS/", which should lead to a Physiological recordings folder, then to a "Matlab Preprocessed Data" folder containing .mat files. <br/>
- Data Name: AMIGOS <br/>
- Channels: 14 <br/>
- Number of Classes: 2 <br/>
  
**DREAMER Setup:** <br/>
- Data Path: Ensure your file path is "your_path_to/DREAMER.mat". <br/>
- Data Name: DREAMER <br/>
- Channels: 14 <br/>
- Number of Classes: 2

**Seed Setup:**
- Data Path: Use the structure "your_path_to/Preprocessed_EEG", which should contain .mat files, a channel-order.xlsx file, and a label.mat file.<br/>
- Data Name: Seed<br/>
- Channels: 62<br/>
- Number of Classes: 3

**SeedIV Setup:**
- Data Path: Ensure your directory structure follows "your_path_to/eeg_raw_data", containing three session folders. Each session folder must include .mat files. The "eeg_raw_data" folder should also contain "Channel Order.xlsx" and "ReadMe.txt" files.<br/>
- Data Name: SeedIV<br/>
- Channels: 62<br/>
- Number of Classes: 4

**[Struture of the framework](https://miro.com/app/board/uXjVMEB2nB0=/?share_link_id=710707650624)**

### **5. Leave-one-subject-out Evaluation Results**
The following table shows the **pre-processing** done on each dataset:

| Dataset     | Cropping                   | Channels Dropped                                                                                      | Band-pass Filter | Notch Filter | Ground Truth |
|-------------|-----------------------------|--------------------------------------------------------------------------------------------------------|------------------|--------------|---------------|
| MAHNOB-HCI  | 30 secs pre and post-baseline | EXG1, EXG2, EXG3, EXG4, EXG5, EXG6, EXG7, EXG8, GSR1, GSR2, Erg1, Erg2, Resp, Temp, Status              | [0.3Hz, 45Hz]     | 50Hz         | ≤ 4.5         |
| DEAP        | 3 secs pre-baseline         | EXG1, EXG2, EXG3, EXG4, GSR1, Plet, Resp, Temp                                                         | -                | 50Hz         | ≤ 4.5         |
| AMIGOS      | -                           | ECG_Right, ECG_Left, GSR                                                                              | -                | 50Hz         | ≤ 4.5         |
| DREAMER     | -                           | -                                                                                                     | [0.3Hz, 45Hz]     | 50Hz         | ≤ 3           |
| SEED        | -                           | -                                                                                                     | [0.3Hz, 45Hz]     | 50Hz         | -             |
| SEED IV     | -                           | -                                                                                                     | [0.3Hz, 45Hz]     | 50Hz         | -             |


All datasets were resampled using a sampling rate of 128Hz. Segments of the signal are created using a window size of 4 with an overlap of 0. All experiments were run for 200 epochs, with a batch size of 32. The learning rate used was 0.001 with no weight decay. For the Cross-entropy loss function, a label smoothing of 0.01 was used as we found that it slightly increased Accuracies (≈1%) for some models. For training the different methods, the data was split by subject into 80% training and 20% validation sets. In each fold of the LOSO loop, we select the model with the best accuracy on the validation set for evaluation on the test subject. To ensure reproducibility, we ran all our experiments using the random seed value of 2025.

(⭐= Best performance)

#### Mahnob Dataset

| Task    | Model            | Accuracy          | F1                | F1 Weighted       |
| ------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Arousal | TSception        | **0.54 ± 0.11** ⭐ | **0.49 ± 0.24** ⭐ | 0.51 ± 0.12       |
|         | EEGNet           | 0.52 ± 0.11       | 0.43 ± 0.21       | 0.49 ± 0.12       |
|         | DeepConvNet      | **0.54 ± 0.13** ⭐ | 0.44 ± 0.27       | 0.49 ± 0.16       |
|         | ShallowConvNet   | 0.52 ± 0.11       | 0.44 ± 0.21       | 0.49 ± 0.12       |
|         | Trivial Baseline | 0.34 ± 0.11       | 0.48 ± 0.11       | **0.52 ± 0.04** ⭐ |
| Valence | TSception        | 0.53 ± 0.07       | 0.52 ± 0.16       | 0.50 ± 0.08       |
|         | EEGNet           | 0.56 ± 0.08       | **0.58 ± 0.15** ⭐ | 0.53 ± 0.10       |
|         | DeepConvNet      | 0.56 ± 0.08       | 0.56 ± 0.18       | 0.53 ± 0.11       |
|         | ShallowConvNet   | **0.57 ± 0.08** ⭐ | 0.57 ± 0.18       | **0.54 ± 0.10** ⭐ |
|         | Trivial Baseline | 0.55 ± 0.10       | 0.53 ± 0.06       | 0.50 ± 0.03       |

#### Deap Dataset

| Task    | Model            | Accuracy          | F1                | F1 Weighted       |
| ------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Arousal | TSception        | 0.54 ± 0.09       | 0.56 ± 0.20       | **0.53 ± 0.10** ⭐ |
|         | EEGNet           | 0.53 ± 0.13       | 0.53 ± 0.21       | 0.50 ± 0.13       |
|         | DeepConvNet      | 0.57 ± 0.14       | 0.52 ± 0.31       | 0.49 ± 0.16       |
|         | ShallowConvNet   | 0.56 ± 0.13       | 0.57 ± 0.26       | 0.49 ± 0.13       |
|         | Trivial Baseline | **0.59 ± 0.15** ⭐ | **0.58 ± 0.08** ⭐ | **0.53 ± 0.04** ⭐ |
| Valence | TSception        | 0.51 ± 0.08       | 0.55 ± 0.16       | 0.47 ± 0.08       |
|         | EEGNet           | 0.53 ± 0.10       | 0.56 ± 0.21       | 0.45 ± 0.12       |
|         | DeepConvNet      | 0.51 ± 0.11       | 0.47 ± 0.28       | 0.41 ± 0.13       |
|         | ShallowConvNet   | 0.53 ± 0.08       | **0.61 ± 0.18** ⭐ | 0.45 ± 0.12       |
|         | Trivial Baseline | **0.57 ± 0.09** ⭐ | 0.56 ± 0.05       | **0.51 ± 0.03** ⭐ |

#### Amigos Dataset

| Task    | Model            | Accuracy          | F1                | F1 Weighted       |
| ------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Arousal | TSception        | 0.58 ± 0.18       | 0.64 ± 0.24       | 0.57 ± 0.20       |
|         | EEGNet           | 0.60 ± 0.23       | 0.69 ± 0.24       | 0.56 ± 0.25       |
|         | DeepConvNet      | 0.56 ± 0.21       | 0.65 ± 0.24       | 0.55 ± 0.22       |
|         | ShallowConvNet   | 0.59 ± 0.23       | **0.70 ± 0.23** ⭐ | 0.55 ± 0.25       |
|         | Trivial Baseline | **0.66 ± 0.26** ⭐ | 0.62 ± 0.19       | **0.59 ± 0.11** ⭐ |
| Valence | TSception        | 0.53 ± 0.10       | 0.56 ± 0.18       | 0.51 ± 0.13       |
|         | EEGNet           | 0.55 ± 0.11       | 0.59 ± 0.19       | 0.50 ± 0.14       |
|         | DeepConvNet      | 0.55 ± 0.11       | 0.56 ± 0.18       | **0.52 ± 0.13** ⭐ |
|         | ShallowConvNet   | 0.55 ± 0.13       | **0.60 ± 0.20** ⭐ | 0.51 ± 0.15       |
|         | Trivial Baseline | **0.56 ± 0.14** ⭐ | 0.55 ± 0.07       | 0.51 ± 0.05       |

#### Dreamer Dataset

| Task    | Model            | Accuracy          | F1                | F1 Weighted       |
| ------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Arousal | TSception        | 0.47 ± 0.07       | 0.43 ± 0.14       | 0.46 ± 0.08       |
|         | EEGNet           | 0.46 ± 0.09       | **0.47 ± 0.19** ⭐ | 0.42 ± 0.13       |
|         | DeepConvNet      | 0.48 ± 0.11       | 0.41 ± 0.18       | 0.45 ± 0.12       |
|         | ShallowConvNet   | 0.48 ± 0.08       | 0.41 ± 0.18       | 0.45 ± 0.10       |
|         | Trivial Baseline | **0.52 ± 0.15** ⭐ | 0.46 ± 0.07       | **0.51 ± 0.02** ⭐ |
| Valence | TSception        | **0.60 ± 0.06** ⭐ | **0.42 ± 0.15** ⭐ | **0.57 ± 0.07** ⭐ |
|         | EEGNet           | **0.60 ± 0.08** ⭐ | 0.26 ± 0.20       | 0.53 ± 0.11       |
|         | DeepConvNet      | **0.60 ± 0.09** ⭐ | 0.38 ± 0.20       | 0.56 ± 0.10       |
|         | ShallowConvNet   | **0.60 ± 0.08** ⭐ | 0.35 ± 0.19       | 0.56 ± 0.10       |
|         | Trivial Baseline | 0.59 ± 0.09       | 0.40 ± 0.04       | 0.51 ± 0.03       |

#### SEED Dataset

| Task        | Model            | Accuracy          | F1                | F1 Weighted       |
| ----------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Categorical | TSception        | 0.48 ± 0.07       | 0.46 ± 0.08       | 0.46 ± 0.08       |
|             | EEGNet           | 0.46 ± 0.06       | 0.44 ± 0.06       | 0.44 ± 0.06       |
|             | DeepConvNet      | **0.55 ± 0.08** ⭐ | **0.52 ± 0.10** ⭐ | **0.52 ± 0.10** ⭐ |
|             | ShallowConvNet   | 0.49 ± 0.06       | 0.47 ± 0.07       | 0.47 ± 0.07       |
|             | Trivial Baseline | 0.34 ± 0.00       | 0.34 ± 0.02       | 0.34 ± 0.02       |

#### SEED IV Dataset

| Task        | Model            | Accuracy          | F1                | F1 Weighted       |
| ----------- | ---------------- | ----------------- | ----------------- | ----------------- |
| Categorical | TSception        | 0.40 ± 0.08       | 0.33 ± 0.10       | 0.35 ± 0.10       |
|             | EEGNet           | 0.32 ± 0.03       | 0.20 ± 0.06       | 0.22 ± 0.05       |
|             | DeepConvNet      | **0.45 ± 0.09** ⭐ | **0.42 ± 0.11** ⭐ | **0.42 ± 0.11** ⭐ |
|             | ShallowConvNet   | 0.37 ± 0.06       | 0.32 ± 0.06       | 0.33 ± 0.07       |
|             | Trivial Baseline | 0.31 ± 0.00       | 0.24 ± 0.03       | 0.24 ± 0.03       |

### **6. Leave-one-trial-out Validation Results**
These results showcase the LOTO experiments that were conducted to replicate the [TSception](https://ieeexplore.ieee.org/document/9762054) paper.

To run the LOTO experiments on DEAP with TSception model, please follow the instruction:-

1. In ```helpers.py``` file, uncomment the four extra channels ("Oz", "Pz", "Fz", "Cz") in DEAPConfig that are dropped in the TSception paper.
2. Comment out the notch filtering as well from DEAPConfig
3. Use the following ```run_cli.sh``` file:
   
```
#!/bin/bash
python run_cli.py \
--model_name=TSception \
--data_name=DEAP \
--data_path='path_to_DEAP/data_preprocessed_python' \
--data_config=DEAPConfig \
--split_type="LOTO" \
--num_classes=2 \
--ground_truth_threshold=5 \
--sampling_r=128 \
--window=4 \
--overlap=0 \
--label_type="A" \
--num_epochs=500 \
--batch_size=64 \
--lr=0.001 \
--weight_decay=0 \
--label_smoothing=0 \
--dropout_rate=0.5 \
--channels=28 \
--train_val_split=0.8 \
--random_seed=2021 \
--log_dir="logs/..." \
--overal_log_file="DEAP_TSception_A_LOTO.txt" \
--log_predictions=True \
--log_predictions_dir="Logged_predictions/DEAP_TSception_A_LOTO" \
```

| **Method**                  | **Arousal ACC** | **Arousal F1** | **Valence ACC** | **Valence F1** |
| --------------------------- | --------------- | -------------- | --------------- | -------------- |
| **SVM**                     | 62.00%          | 58.30%         | 57.60%          | 56.30%         |
| **UL**                      | 62.34%          | 60.44%         | 56.25%          | 61.25%         |
| **CSP**                     | 58.26%          | --             | 57.59%          | --             |
| **FBCSP**                   | 59.13%          | --             | 59.19%          | --             |
| **FgMDM**                   | 60.04%          | --             | 58.87%          | --             |
| **TSC**                     | 60.04%          | --             | 59.47%          | --             |
| **FBFgMDM**                 | 60.30%          | --             | 61.01%          | --             |
| **FBTSC**                   | 60.60%          | --             | 61.09%          | --             |
| **TSception**               | 63.75%          | 63.35%         | 62.27%          | 65.37%         |
| **TSception (ours)**        | 60.67%          | 61.40%         | 59.32%          | 62.49%         |
| **Trivial Baseline (ours)** | 62.73%          | 55.82%         | 50.31%          | 53.81%         |


### **7. License**:
This code repository is licensed under the [CC BY 4.0 License](LICENSE).

### **8. Citation**: