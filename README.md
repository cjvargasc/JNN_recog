# Joint Neural Networks for One-shot Object recognition

This is the implementation of the "Joint Neural Networks for One-shot Object Recognition and Detection" thesis by Camilo Vargas.

## Requirements
* Python 3.x
* Numpy
* OpenCV
* Pytorch
* Matplotlib
* PIL
* QMUL-OpenLogo dataset (https://qmul-openlogo.github.io/)
* MiniImagenet dataset (https://github.com/yaoyao-liu/mini-imagenet-tools)

## Usage
Set the training and testing parameters in ```python params/config.py``` file. Run the ```python main.py``` file to train/test the defined configuration.

## Results

Accuracy in MiniImagenet compared to previous methods

Method CNN | Accuracy(%)
-------------|-----|
MALM | 48.70
ProtoNets | 49.42
SNAIL | 55.71 
TADAM | 58.50 
MLT | 61.20
Proposed | 61.41

## Reference
Joint Neural Networks for One-shot ObjectRecognition and Detection. Camilo Jose Vargas Cortes. School of Electronic Engineering and Computer Science. Queen Mary University of London. 2020.

## Examples
(Showing the resulting similarity metric [0,1] for each pair of images)

<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/3m1.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/3m5.png" width="25%">
S = 0.63

<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/3m1.png" width="25%">-<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/ANZ_sportslogo_1.png" width="25%">
S = 0.13

<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/n0327201000000011.jpg" width="25%">-<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/n0327201000000013.jpg" width="25%">
S = 0.62

<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/n0327201000000011.jpg" width="25%">-<img src="https://github.com/cjvargasc/JNN_recog/blob/master/imgs/n0452216800000088.jpg" width="25%">
S = 0.23
