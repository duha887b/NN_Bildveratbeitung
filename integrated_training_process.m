%% Load Training Data & define class catalog & define input image size
disp('Loading training data...')
% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid

%@Dustin Hanusch 

oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
ImagesTrain = 'train-images-idx3-ubyte.gz';
LabelsTrain = 'train-labels-idx1-ubyte.gz';
ImagesTest = 't10k-images-idx3-ubyte.gz';
LabelsTest = 't10k-labels-idx1-ubyte.gz';

X1 = processImagesMNIST(ImagesTrain);
Y1 = processLabelsMNIST(LabelsTrain);
X2 = processImagesMNIST(ImagesTest);
Y2 = processLabelsMNIST(LabelsTest);
XImages = X1;
YLabels = Y1;
%TODO combine 2 arrays

[trainInd,testInd,validInd] = dividerand(numel(YLabels),0.7,0.1,0.2);

XTrain = XImages(trainInd);
YTrain = YLabels(trainInd);
XTest = XImages(testInd);
YTest = YLabels(testInd);
XValid = XImages(validInd);
YValid = YLabels(validInd);

path(oldpath);

%% define network
%   most basic network
%NN_layers = [ ...

%];

% visualize the neural network
%analyzeNetwork(NN_layers)
%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 
% solver "sgdm" "rmsprop" "adam"

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%%  Train neural network
% define "trainingOptions"
% training using "trainNetwork"



%% test neural network & visualization 
% Calculate accuracy

