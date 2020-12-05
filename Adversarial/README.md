# Introduction to whitebox Adversarial Exemples

## Description

This directory contains two files: 
 - Lecture slides: 00_Main_deep_robust.pdf
 - Associated notebook: adversarial_attacks.ipynb. This notbook requires pytorch, [cvxpy](https://www.cvxpy.org/) and gurobi (if not available, it can be replaced in the code by 'ECOS_BB')


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

<p align="center"> <img src="https://www.cs.umd.edu/~tomg/img/free/viz_9985_10000_small.png" 
alt="CIFAR10 data" width="540" height="270" border="1"  class="center" /> <br/>
<i>Illustration from the Adversarial Training for Free! web site </i> https://www.cs.umd.edu/~tomg/projects/free/
</p>
 

For more details, see for instance [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org/), 
the [Adversarial Machine Learning Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html) 
or the [RobustBench web site](https://robustbench.github.io/) whose  goal is to systematically track the real progress in adversarial robustness.


## A list of review papers

 - Akhta et al, Threat of Adversarial Attacks on Deep Learning 
in Computer Vision: A Survey (IEEE acces, feb 2018)
https://ieeexplore.ieee.org/document/8294186

 - Chakraborty et al, Adversarial Attacks and Defences: A Survey (sept 2018)
https://arxiv.org/abs/1810.00069

 - Biggio \& F Roli, Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning
Pattern Recognition (dec 2018)
https://www.sciencedirect.com/science/article/abs/pii/S0031320318302565

 - Yuan et al, Adversarial examples: Attacks and defenses for deep learning
 IEEE transactions on neural networks, (jan 2019)
 https://arxiv.org/abs/1712.07107

 - Xu et al, Adversarial Attacks and Defenses in Images, Graphs and Text: A Review (sept 2019)
https://arxiv.org/abs/1909.08072

 - Wiyatno et al., Adversarial Examples in Modern Machine Learning:  A Review (nov 2019)
https://arxiv.org/pdf/1911.05268.pdf

 - Silva \& Najafirad,  Opportunities and Challenges in Deep Learning
Adversarial Robustness: A Survey (jul 2020)
https://arxiv.org/abs/2007.00753


