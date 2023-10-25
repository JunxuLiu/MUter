# Muter: Machine Unlearning on Adversarially Trained Models

Code for the ICCV'23 paper [MUter: Machine Unlearning on Adversarially Trained Models](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_MUter_Machine_Unlearning_on_Adversarially_Trained_Models_ICCV_2023_paper.html).

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

