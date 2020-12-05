# Introduction to whitebox Adversarial Exemples

## Description

This directory contains two files: 
 - Lecture slides: 00_Main_deep_robust.pdf
 - Associated notebook: adversarial_attacks.ipynb. This notbook requires pytorch, cvxpy and gurobi (if not available, it can be replaced in the code by 'ECOS_BB')


## How to build adversarial exemples and define the robustness of a classifier?

Recent results have shown that neural networks are vulnerable to so-called adversarial attacks, that is, subtle input disturbances that may be too small to be noticeable, but nevertheless capable of fooling deep learning models. 
These slides aim at providing an introduction to the notion of adversarial exmples in deep learning, focussing on how to build them.  
To practice, a Jupyter Notebook is also available.

This notebook illustrate through simmple implementations of main methods to generate such adversarial examples in a white box setting, with the CIFAR10 dataset using a 3-layer perceptron as a neural network implemented with pytorch, in order to compare their efficiency both in terms of precision and computation time.

Are implemented:
- Evasion attack
- Fast Gradient Sign Method attack
- Projected Gradient Method attack
- Carlini & Wagner attack
- The MIP-ReLUplex attacks using either L_2 and L_&infin; distance measures
- The accelerated MIP-ReLUplex attacks using bounds

<center>
 <img src="https://www.cs.umd.edu/~tomg/img/free/viz_9985_10000_small.png" 
alt="CIFAR10 data" width="470" height="270" border="1"  class="center" />
<p style="text-align: center;"> <i>Illustration from the Adversarial Training for Free! web site </i> https://www.cs.umd.edu/~tomg/projects/free/</p>
 </center>
 

For more details, see for instance [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org/)
or the [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)
