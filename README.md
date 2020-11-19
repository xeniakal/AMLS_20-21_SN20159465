# AMLS_20-21_SN20159465

This assignment involves the realisation of various machine learning tasks on two provided datasets that are subsets of the original CelebFaces Attributes Dataset (CelebA) and Cartoon Set. The datasets can be downloaded via following link: https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view?usp=sharing. 

We went through the data, analysed it and pre-processed it as necessary. We designed separate modes for each task and report training errors, validation errors, hyper-parameter tuning. To choose the best-fitted model we tried several models for each task, as you can see in code files comparing the results through tables and plots.

## Introduction

This repository is part of the final assignment of AMLS 2020-2021 course of Integrated Machine Learning Systems MSc at UCL. It consists of four tasks **A1, A2, B1, B2** which are explained on [Tasks](#tasks).
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
  - README.md
  
The **A1, A2, B1 and B2** folders contain the code files for each task, besides the pretrained models that showed the better performance.
* In *.ipynb* files we present all steps taken to solve the tasks, explaining our models and design choices. In addition, we explain briefly the results obtained via our experiments and provide accuracy prediction scores on unseen data (Test Set). To compare the efficiency of each model we present their metrics and compare them through plots and tables.
* In *.py* files, only final model's code is included with the best preprocessing technique for each task, while accuracy reports (including metrics and confusion matrixes) for each train, validation and test data are provided.
* **‘Dataset’** folder is for the shake of memory bandwidth on Github. If you want to run the project use this folder to insert the datasets from the link provided.

## Tasks

Add content...
(say what .ipynb and .py do)


## Workflow graph

Add pipeline procedures...

## Dependencies

This repo was tested on `Python 3.6.12`. You are encouraged to use this version or later.

If you have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster.

The project was implemented in a windows machine, on conda environment, a friendly interface to run python projects. Python Notebooks were executed on Jupyter Notebook, while python files on Conda integrated environment. Before installing anything, it is suggested to create a conda environment which will satisfy our project dependencies.

If you want to run this repository, first install the required dependencies:

```
git clone https://github.com/xeniakal/AMLS_20-21_SN20159465.git
pip install -r requirements.txt
```

All neccessary libraries are imported through `imports.py` file. Just import this file in your code and all dependencies will be inserted automatically.

## Code and Folders

Explain how to run...



## Final results

Add table with final accuracy results
