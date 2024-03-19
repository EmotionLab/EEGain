# **EEGain**

## Run experiments for EEG-based emotion recognition for most popular datasets and models with just one command

### **Description**
EEG-based emotion recognition has become increasingly popular direction in recent years. Although lots of researchers are working on this task, running experiments is still very difficult. The main challenges are related to dataset and model. Running experiment on new dataset means that researcher should implement it with pytorch or tensorflow, get detailed information how dataset is recorded and saved and many more. Running experiment on new model is also tricky, researcher should implement it from scratch with pytorch or tensorflow. This process on one hand takes too much time and can cause lots of bugs and effort, on the other hand it is not helpful for researcher for further research. To solve this problem, make process easier and get more researchers in this field, we created EEGain, is a novel framework for EEG-based emotion recognition to run experiments on different datasets and models easily, with one command. You can implement your custom models and datasets too. 

Models that are implemented in EEGain for now - EEGNet, TSception, DeepConvNet and ShallowConvNet.

Datasets that are implemented in EEGain for now - DEAP, MAHNOB, AMIGOS, DREAMER, Seed and SeedIV.

Some other models and datasets are comming. 

### **QuickStart**
to run framework quickly and test it, you can use this colab notebook - **[link](https://colab.research.google.com/drive/1msoWvWbY_Ztrb2ny0SpibyaAbPKWp-9Q?usp=sharing)** 

You just need to clone this repo and run .sh file with proper parameters. 
You can see results on tensorboard. 

### **How to run**
1. clone the repo
2. Enter in EEGain folder. run <code>pip install .</code>
3. change data path in **main.py**
4. run <code> python run_*.py </code>
5. to check the results: <br>
   - run <code>tensorboard --logdir ./logs </code>
   - logs will be saved in **log.log** file as well<br> 

**[Struture of the framework](https://miro.com/app/board/uXjVMEB2nB0=/?share_link_id=710707650624)** 

### **Processing Time for Different Datasets on Various Models**:

When running different models (such as **DeepConvNet**, **ShallowConvNet**, **EEGnet**, and **Tsception**) on a subject-dependent scenario, the time it takes to process the data can vary a little, but it primarily depends on the specific dataset being used.

Our tests were conducted on Google Colab, utilizing a V100 GPU, with a batch size of 32 and a total of 30 epochs for each run. The following is an estimate of the time required to complete the entire pipeline, using a Leave-One-Trial-Out (LOTO) split, for each dataset:

**DREAMER** Dataset: Approximately 2 to 3 hours.

**DEAP** Dataset: About 2.5 to 3.5 hours.

**MAHNOB** Dataset: Roughly 2 to 3 hours.

**AMIGOS** Dataset: Around 5 to 6 hours.
