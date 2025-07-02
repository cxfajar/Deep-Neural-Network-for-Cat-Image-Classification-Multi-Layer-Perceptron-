# Deep Neural Network for Image Classification: Application

This Jupyter Notebook is the capstone assignment for Week 4 of the **Deep Learning Specialization** (Course 1: Neural Networks and Deep Learning) by DeepLearning.AI. The project builds a **deep neural network (L-layer model)** to classify images as cat or non-cat using the full architecture of a feedforward neural network implemented from scratch.

---

## 📘 Overview

You will use helper functions from previous assignments to build a multi-layer neural network with forward propagation, cost calculation, backward propagation, and parameter updates.

The model will be trained on a labeled dataset of cat and non-cat images, and tested for accuracy. You'll compare the performance of a simple 2-layer neural network versus a deeper L-layer network, showing how depth can increase classification performance on complex datasets.

---

## 🧠 Learning Outcomes

By completing this assignment, you will:

* Implement a **2-layer neural network** and a **deep L-layer neural network**
* Understand and apply:

  * Forward propagation through multiple layers
  * Backward propagation through deep networks
  * Vectorized implementations for scalability
* Use **cross-entropy cost function** for binary classification
* Optimize weights and biases using **gradient descent**
* Train and evaluate the model using image data
* Test the model with your own images

---

## 🧪 Notebook Contents

### 📦 Section 1: Importing Packages

Basic scientific libraries like NumPy, matplotlib, h5py, and utility functions from `dnn_app_utils`.

### 🖼️ Section 2: Load and Process Dataset

* Use a cat/non-cat labeled dataset (64x64 images)
* Reshape and normalize data

### 🏗️ Section 3: Model Architectures

* Define 2-layer and L-layer model structures
* Compare shallow vs deep networks

### 🧠 Section 4: Two-layer Neural Network

* Implement `two_layer_model(X, Y, ...)`

### 🔁 Section 5: L-layer Deep Neural Network

* Implement `L_layer_model(X, Y, ...)` with arbitrary number of layers

### 📊 Section 6: Performance Evaluation

* Compute train/test accuracy
* Plot learning curves

### 🖼️ Section 7: Test with Your Own Image

* Upload a custom image and make a prediction using the trained model

---

## 🖥️ Requirements

Install the following Python libraries:

```bash
pip install numpy matplotlib h5py scipy pillow
```

Also, include the `dnn_app_utils.py` file in the same directory as the notebook.


---

## 🎯 Who Should Use This

* Machine learning and AI learners exploring deep learning concepts
* Anyone who completed Course 1 of the DeepLearning.AI specialization
* Engineers and students wanting to understand how L-layer neural networks are constructed from scratch

---



## 🧪 Final Outcome

You will build a deep learning classifier that can distinguish cat vs non-cat images, achieving high accuracy on both training and test datasets and extending your previous work with logistic regression and shallow networks.

---

