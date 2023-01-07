
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Next to the Anaconda distribution of Python you need the following packages/librariers to execute the project.
- pandas
- numpy
- sklearn
- sqlalchemy
- seaborn
- pickle
- matplotlib

The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I used data from Kaggle (https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) that contains synthetic times series machine data and a categorization whether there was a failure or not. In addition, it specifies what kind of failure was experienced. I was interested in encoding categorical data and training a supervised machine learning model. Following that, the trained model is able to categorize new data into the known failure categories. This helps to predict the failure and type of failure for future processes of the machine.

## File Descriptions <a name="files"></a>

The work is separated into the folders data, model and blogpost. 
In the data folder, the needed input data predictive_maintenance.csv, the process_data.py and the visualize_data.py can be found. The process_data.py extracts, tranforms and loads the data. It loads the data into a database PredictiveMaintenance.db. The visualize_data.py cretaes different visualization of the data to get a better understanding of the data.
In the model folder, the train_classifier.py loads the cleaned data from the database and creates a model which is trained. It will then later extract the model als a pickle file.
The blogpost folder contains the results of the project in an pdf format.

## Results<a name="results"></a>
The results are represented in the blogpost that can be found in the blogpost folder. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data is used from Kaggle (https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

