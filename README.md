**EEGain**

**How to run**
1. clone the repo
2. Enter in EEGain folder. run <code>pip install .</code>
3. change data path in **main.py**
4. run <code> python run_*.py </code>
5. to check the results: <br>
   - run <code>tensorboard --logdir ./logs </code>
   - logs will be saved in **log.log** file as well<br> 

**[Struture of the framework](https://miro.com/app/board/uXjVMEB2nB0=/?share_link_id=710707650624)** <br>
   

**Processing Time for Different Datasets on Various Models**:<br>

When running different models (such as **DeepConvNet**, **ShallowConvNet**, **EEGnet**, and **Tsception**) on a subject-dependent scenario, the time it takes to process the data can vary a little, but it primarily depends on the specific dataset being used.<br>

Our tests were conducted on Google Colab, utilizing a V100 GPU, with a batch size of 32 and a total of 30 epochs for each run. The following is an estimate of the time required to complete the entire pipeline, using a Leave-One-Trial-Out (LOTO) split, for each dataset:<br>

**DREAMER** Dataset: Approximately 2 to 3 hours.<br>
**DEAP** Dataset: About 2.5 to 3.5 hours.<br>
**MAHNOB** Dataset: Roughly 2 to 3 hours.<br>
**AMIGOS** Dataset: Around 5 to 6 hours.<br>
