![Model Interpreter diagram](https://raw.githubusercontent.com/AndreCNF/model-interpreter/master/docs/images/ModelInterpreterMap.png)

---

**Model Interpreter** is a package for intuitive interpretation of machine learning models, with an emphasis on multivariate time series an recurrent neural networks settings.

## Install

There are two ways to install Model Interpreter:

* Clone the repository **(Recommended)**

  In order to install this repository with all the right packages, you should follow this steps:

  1. Clone through git

  ```
  git clone https://github.com/AndreCNF/model-interpreter.git
  ```

  2. Install the requirements

  If you have [Poetry](https://python-poetry.org/) installed, which is recommended, you can just enter the directory where the code has been cloned to and run the following command, which will create a new virtual environment and install all the requirements:

  ```
  poetry install
  ```

  Otherwise, you can just run:

  ```
  pip install -r requirements.txt
  ```

* Install through pip

  Using pip, you can install with the following command:
  `pip install -e git+https://github.com/AndreCNF/model-interpreter.git`

## Overview

Model Interpreter is meant to facilitate all the interpretability procedure, through a Model Interpreter class that can handle the intermediate steps to achieve the importance scores, the data and visualizations that we desire. After initializing a Model Interpreter object with the model and the data that we want to use in our analysis, we gain access to the `interpret_model` method. This Python function allows us to calculate feature importance, through the integration of a [custom SHAP](https://github.com/AndreCNF/shap), instance importance, which is a new technique that complements SHAP on multivariate time series, or both at the same time.

## Feature importance on RNNs

This package integrates a [custom version of SHAP](https://github.com/AndreCNF/shap), which adapts its robust, perturbation-based feature importance techniques to Recurrent Neural Networks (RNNs), even if these are bidirectional. The core modification lies on added parameters that identify the model type, which in the case of it being RNN-based changes the pipeline to maintain the model's hidden memories across each sequence. For more details, check the paper referenced on the bottom of the page.

## Instance importance score

SHAP can only calculate feature importance scores, i.e. the influence of each feature on a given output, which does not complete the interpretation pipeline, which again misses some extra care when addressing multivariate time series. So, Model Interpreter includes an instance importance score, which allows to interpret how each instance in a given time series impacts the sequence's final output. This score is defined as:

![Instance importance equation](https://raw.githubusercontent.com/AndreCNF/model-interpreter/master/docs/images/InstanceImportanceEq.png)

It essentially combines instance occlusion with an output variation metric in a weighted sum. For more details, check the paper referenced bellow.

Having this instance importance formulation, we can visualize the scores, even in multiple sequences simultaneously. Model Interpreter has a visualization method for this, `instance_importance_plot`, inspired by the [RetainVis paper](https://arxiv.org/abs/1805.10724) from Bum Chul Kwon et. al, in which we can see each patient's time series and their clinical visits, colored according to their impact on the final output (red indicates an positive impact in the output and blue indicates a negative impact).

![Patients time series](https://raw.githubusercontent.com/AndreCNF/model-interpreter/master/docs/images/PatientsTimeSeries.png)

## Paper and citation

For more information on how Model Interpreter works and its motivation, you can check my [master's thesis](http://andrecnf.github.io/master-thesis).

If you want to include Model Interpreter on your citations, please use the following .bib file:

```
I will put here the citation when I have the paper published :)
```
