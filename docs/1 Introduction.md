# Introduction to Machine Learning and Data Generation
### Applications (general)
- Image recognition
- Voice recognition
- Natural language processing
- Artificial intelligence
- Generative adversarial networks
- Self-driving cars
- Applications (specific)
- DNA sequencing
- Read translation
- Taxonomic classification
- Blood tests (WBIT)
- Danny’s physics experiment

For an idea of where we are heading in this section of the course, visit the site https://playground.tensorflow.org/ which allows you to build your own neural networks and use them to classify several classic datasets. The course will be very hands on, we will do everything from generating the classic datasets ourselves, through to coding a simple logistic classifier before finally learning about the most basic neural network type: multi-layer perceptrons. Along the way we will learn several key machine learning concepts including regularization, feature engineering, batch training, dropout, etc. 

##Data Generation

Although machine learning is much more general and powerful, initially we will use classification tasks to introduce ourselves to many of the key concepts. In doing so, we will see that classification tasks essentially amount to a geometry problem: finding surfaces in our ‘n-feature’-dimensional space that separate our groups into separate classes. As a corollary we will see that a common theme runs through all of the classification tasks that we are going to look at: all data is the same.

## Weekly tasks

As a warm-up task (i.e. before we introduce any machine learning-specific elements) we will first refresh our numpy skills by writing a program capable of generating several of the classic machine learning datasets (see https://playground.tensorflow.org/).

To begin with, we will focus on binary classification tasks where our targets are assigned the binary variable: 0 or 1. However, since the eventual plan will be to look at multinomial problems, we will use this initial opportunity to generalize our data generator program to produce datasets with an arbitrary number of target classes: `n_targets`.

The skeleton code (i.e. the code containing the class delcarations/methods/help text but with the actual function definitions and algorithm's absent) for this week's task can be found in the script data.py.

#### `DataGenerator`

- Write a Python program (class) called `DataGenerator` to generate the cloud, donut, xor and spiral datasets (see the help file provided).

#### `DataContainer`

- Create another class called `DataContainer` that inherits the methods of `DataGenerator` and adds a new method `plot()` to allow us to visualize the data.

### Notes
- Test datasets can be found at: https://playground.tensorflow.org/
- How to document your code: https://realpython.com/documenting-python-code/