# Muter: Machine Unlearning for Adversarial Traning Models

## Dependencies

    torch, torchvision, sklearn, functorch

## Setup
Project structure

    MUter_code/
    -->Readme
    -->setup
    -->LinearExperiment
        -->data
        -->src
    -->NNExperiment
        -->src
            -->data

run script setup.sh for complete project structure, download source data and do data process.
    
    sh setup.sh

## Run Linear Experiment

    cd LinearExperiment/src/
    sh run_shell.sh


## Run Neural Network Experiment

    cd NNExperiment/src/
    sh sh_code.sh

# Junxu's modifications

1) construct duplicated code as functions so making the code more concisely.

2) set the random seed for repeated experiments

3) save the atm models and training time (combined as a dict) of each conditions for avoiding time wastes when we need to test the same conditions again. Note that condistions with all same augments expect `args.times` should be seem as the different runs.

