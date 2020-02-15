# Visual Concept Learning

This repository contains code for studying object recognition in humans. It
specifically contains a set of simulations in a computational model of object
recognition which show that it is useful to leverage pre-existing high-level
visual features representations (i.e. concept-level features) when learning new
visual concepts.

## Installation

1. Install external tools as needed. These simulations rely on a number of
   external tools. The tools are listed below, along with the versions I've
   tested:
   - caffe (master branch, commit fe0f441)
   - MATLAB (2015b)
   - python 2 (2.7.9)
   - WordNet (3.0)

2. Clone the repo and its submodules:

   ```bash
   git clone --recurse-submodules https://github.com/joshrule/visual-concept-learning.git
   ```

## Usage

From the `src` directory, run:

```bash
# Generate the feature data and set up the binary classification problems.
matlab -nodesktop -nodisplay -nosplash -r "simulation(params()); exit;"

# Run the binary classifications for the main evaluation.
./parallel_wrapper_script.sh evaluation 1 5

# Run the binary classifications for the categoricity analysis.
./parallel_wrapper_script.sh categoricity 1 5

# Compile the results of the main evaluation.
matlab -nodesktop -nodisplay -nosplash -r "compileResults(params(),'googlenet-binary');exit;"
```
