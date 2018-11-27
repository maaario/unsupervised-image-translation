# unsupervised-image-translation
A re-implementation of a paper which uses graphical models for transferring style between images
as my final project for course Graphical Models in Machine Learning, spring 2017. Tweaked and polished in 2018.

[Unsupervised Image Translation, Rómer Rosales, Kannan Achan, and Brendan Frey, 2003](http://people.csail.mit.edu/romer/papers/RosalesAchanFrey_ICCV03.pdf)

Here I implement:
  - loopy belief propagation, approximate inference algorithm using message passing in C++
  - expectation-maximization algorithm as described in the paper
  - a commandline tool for transferring style of one image to another
  - a set of reproducible experiments to observe convergence and effects of some paramters

![Example output](https://github.com/maaario/unsupervised-image-translation/blob/master/example/5.png)

More detailed results of this implementation, can be found in [`example`](https://github.com/maaario/unsupervised-image-translation/blob/master/example) directory.

## How to run it
```
cd unsupervised-image-translation

# Create a virtual environment
virtualenv -p python3 env
. env/bin/activate

# Install necessary modules
pip install -r requirements.txt

# Compile loopy belief propagation written in C++
python setup.py install

# Run
python src/unsupervised_image_translation/main.py -input=inputs/ramona-color.png -source=inputs/starry-night.png

# To list all arguments
python src/unsupervised_image_translation/main.py -h

# Optionally reproduce experiments
bash run_all_experiments.sh
```
