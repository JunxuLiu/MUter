#!/bin/bash

## create file structure

linear_data_dir="LinearExperiment/data"
linear_data_matrix_dir="LinearExperiment/data/MemoryMatrix"
linear_data_COVTYPE_dir="LinearExperiment/data/COVTYPE"

NN_data_dir="NNExperiment/src/data"
NN_data_model_dir="NNExperiment/src/data/model"
NN_data_preMatrix_dir="NNExperiment/src/data/preMatrix"


mkdir $linear_data_dir
mkdir $linear_data_matrix_dir
mkdir $linear_data_COVTYPE_dir

mkdir $NN_data_dir
mkdir $NN_data_model_dir
mkdir $NN_data_preMatrix_dir


## download data

## Linear mdoels data download

# for Covtype
if [ -f "LinearExperiment/data/COVTYPE/covtype.libsvm.binary.scale.bz2" ];then
    echo "file: covtype.libsvm.binary.scale.bz2 exists"
    else
    wget -P LinearExperiment/data/COVTYPE https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
fi


## NN model data download 

## for case: from core dataset downsampled ImageNet to target dataset CIFAR-10, we use the pretrained model from https://arxiv.org/abs/1901.09960, which give a open resource link to download the model, here we download following this way, if download failure, please download manually (imagnet_wrn_baseline_epoch_99.pt) int file <NNExperiment/data/model/pretrain_model>.

## step 1
pip install gdown
## step 2 save pretrain model path to <NNExperiment/data/model/pretrain_model>
gdown https://drive.google.com/drive/folders/10raR7I1hjOl3nuxz1O2yKMzV1jC77r8a -O NNExperiment/data/model/pretrain_model --folder



## data process

# for linear data process
cd LinearExperiment/src/
python dataloder.py

cd ..
cd ..

cd NNExperiment/src/
python data_utils.py

