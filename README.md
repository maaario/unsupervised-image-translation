# unsupervised-image-translation
A re-implementation of a paper which uses graphical models for transferring style between images
as my final project for course Graphical Models in Machine Learning, spring 2016/2017.

[Unsupervised Image Translation, Rómer Rosales, Kannan Achan, and Brendan Frey, 2003](http://people.csail.mit.edu/romer/papers/RosalesAchanFrey_ICCV03.pdf)

Here I implement:
  - loopy belief propagation, approximate inference algorithm using message passing in C++
  - expectation-maximization algorithm as described in the paper
  - a commandline tool for transferring style of one image to another
  - a set of reproducible experiments to observe convergence and effects of some paramters

To see results of this implementation, go to `example` directory.

## How to run it
```
cd unsupervised-image-translation

# Create a virtual envirnoment.
virtualenv -p python3 env
. env/bin/activate

# Install necessary modules.
pip install -r requirements

# Compile loopy belief propagation in C++
python setup.py install

# Run
python src/unsupervised_image_translation/main.py -input=../../inputs/ramona-color.png -source=../../inputs/starry-night.png

# To list all arguments
python src/unsupervised_image_translation/main.py -h

# Optionally reproduce experiments
bash run_all_experiments.sh
```
