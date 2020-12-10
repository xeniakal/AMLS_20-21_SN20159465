# AMLS_20-21_SN20159465

This repository is part of the final assignment of AMLS 2020-2021 course of Integrated Machine Learning Systems MSc at UCL.
It involves the realisation of various machine learning tasks on two provided datasets that are subsets of the original CelebFaces Attributes Dataset (CelebA) and Cartoon Set. The datasets can be downloaded via following link: https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view?usp=sharing. 

We went through the data, analysed it and pre-processed it as necessary. We designed separate modes for each task and reported training errors, validation errors and hyper-parameter tuning. To choose the best-fitted model we tried several models for each task, as you can see in code files, comparing the results through tables and plots.

## Introduction

This assignment consists of four tasks **A1, A2, B1, B2** which are explained on [Tasks](#tasks).
The structure of project is the following:
- AMLS_20-21_SN20159465
  - A1
  - A2
  - B1
  - B2
  - Datasets
    - cartoon_set
    - celeba
    - cartoon_set_test
    - celeba_test
  - main.py
  - imports.py
  - requirements.txt
  - README.md
  
The **A1, A2, B1 and B2** folders contain the code files for each task, besides the pretrained models that showed the better performance. Each folder contains one `.ipynb` file, one `.py` file and the final pretrained model (*pickle or hdf5 format*) which is explained how it was adopted in the corresponding .ipynb file
* In *.ipynb* files we present all steps taken to solve the tasks, explaining our various models and design choices. In addition, we explain briefly the results obtained via our experiments and provide accuracy prediction scores on unseen data (Test Set), besides Train and Dev Set. To compare the efficiency of each model we present their metrics and compare them through plots and tables.
* In *.py* files, only final best performing model's code is included with the best preprocessing technique for each task, while accuracy reports (including metrics and confusion matrixes) for each train, validation and test data are provided.
* **‘Dataset’** folder is empty for the shake of memory bandwidth on Github. If you want to run the project use this folder to insert the datasets from the link provided.
 
**! Notice !**<br/> 
*.ipynb* files contain the whole task pipeline with the multiple models tested until find the best performing!<br/>*.py* files contain only the best performing model - if want to run any other model just follow the headlines inside notebooks to run your desired model.  

## Tasks

In order to avoid overfitting (our model not to be tuned to the data its been given), each time we were running our models, ***a random partition of dataset into train, validation and test set*** was implemented, always in proportion of ***70:15:15***.

* **Task A1**: A classification problem of gender recognition (male/female) on Celeba Dataset.
* **Task A2**: A classification problem of emotion detection (smiling/not smiling) on Celeba Dataset.
* **Task B1**: A multiclass classification problem of Face shape recognition (5 types of face shapes) on Cartoon Set.
* **Task B2**: A multiclass classification problem of Eye color recognition (5 types of eye colors) on Cartoon Set.


## Workflow graph

**The Pipeline procedure:**

<img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/flow.png" height="30%" width="40%">

## Dependencies

This repo was tested on `Python 3.6.12`. You are encouraged to use this version or later.

If you have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster.

The project was implemented in a windows machine, on conda environment, a friendly interface to run python projects. Python Notebooks were executed on Jupyter Notebook, while python files on Conda integrated environment. Before installing anything, it is suggested to create a conda environment which will satisfy our project dependencies.

If you want to run this repository, first install the required dependencies:

```
$ git clone https://github.com/xeniakal/AMLS_20-21_SN20159465.git
$ pip install -r requirements.txt
```

All neccessary libraries are imported through `imports.py` file. Just import this file in your code and all dependencies will be inserted automatically.

## Code Usage

* `.ipynb` files can be run cell by cell, after installing the requirements, in case you want to check every tecnhique and model of our experiments, or just run cells of specific models in which you will be guided throught the arranging of our notebooks.
  * Every jupyter notebook comprises 5 initial steps, besides the next more specified sections that differs in each Task according to the problem:
    * Imports
    * Functions
    * Read Data
    * Baseline Classifiers
    * Preproccessing Data
    * ...
  * Always run first **"Imports"** and **"Functions"** headers
  * To train by yourself the **<ins>final best performing model</ins>** just run the last cell under the subsection **"Train from scratch the best performing model"** under the header **"Best performing model"**.

* `.py` files: To run each task insert as argument your preferred task to the main function. Possible arguments are: `A1` or `a1`,`A2` or `a2`,`B1` or `b1`,`B2` or `b2`, otherwise leave it blank with no argument to execute all tasks, eg:

  `$ python main.py` to run all tasks
  
  `$ python main.py A1` to run A1 task


## Final results

We present our best models' performance for each task on train, validation and test set:

### 1. Table with Final Accuracy Results:

Task | Model | Train Ac | Val Acc | Test Acc
------------ | ------------- | ------------- | ------------- | -------------
A1 | CNN-SVM | 0.94743 | 0.91867 | 0.91733
A2 | CNN | 0.97300 | 0.88000 | 0.89467
B1 | SVM | 1 | 1 | 1
B2 | LogReg | 0.90043 | 0.84267 | 0.84533

### 2. Comparison of Test Set performance on Baseline model vs Best model:

  **Note:** *Check each task's Notebook for further details on the structure and hyperparameters of models*
  
  * **Task A1:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/A1.jpg" height="70%" width="60%">
  
  * **Task A2:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/A2.jpg" height="70%" width="60%">
  
  * **Task B1:**<br/> 
  *Further details of this unusual results on Task's B1 Notebook...*
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/B1.jpg" height="40%" width="40%">
  
  * **Task B2:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/B2.jpg" height="70%" width="60%">
  
### 3. Let's play with the images..

  * **Task A1:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/A1img.jpg">
  
  * **Task A2:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/A2img.jpg">
  
  * **Task B1:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/B1img.jpg">
  
  * **Task B2:**
  
  <img src="https://github.com/xeniakal/AMLS_20-21_SN20159465/blob/master/images/B2img.jpg">
  
  
  
  
  
