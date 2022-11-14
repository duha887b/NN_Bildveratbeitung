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

path(oldpath);

%TODO combine 2 arrays
XImages = X1;
YLabels = Y1;


[trainInd,testInd,validInd] = dividerand(numel(YLabels),0.7,0.1,0.2);

XTrain = XImages(:,:,:,trainInd);
YTrain = YLabels(trainInd);
XTest = XImages(:,:,:,testInd);
YTest = YLabels(testInd);
XValid = XImages(:,:,:,validInd);
YValid = YLabels(validInd);

% zweiter Verusch 
%% define network
%   most basic networklayer0

inputSize = [28 28 1];
layer0 = imageInputLayer(inputSize);

outputSize = 392;
layer1 = fullyConnectedLayer(outputSize);

layer2 = reluLayer;

layer3 = fullyConnectedLayer(10);

layer4 = softmaxLayer;

layer5 = classificationLayer;

NN_layers = [ ...
    layer0
    layer1
    layer2
    layer3
    layer4
    layer5
    ];

%visualize the neural network
analyzeNetwork(NN_layers)
%% Specify Training Options (define hyperparameters)

% solver "sgdm" "rmsprop" "adam"
options = trainingOptions("adam");

% miniBatchSize
options.MiniBatchSize = 128;
% numEpochs
options.MaxEpochs = 30;
% learnRate
options.InitialLearnRate = 0.001;
% executionEnvironment
options.ExecutionEnvironment = 'auto';
% numIterationsPerEpoch 
%options.numIterationsPerEpoch = 5 ;

options.OutputNetwork = 'best-validation-loss';

options.ValidationData = {XValid, YValid};

options.Plots = 'training-progress';


%%  Train neural network
% define "trainingOptions"
% training using "trainNetwork"
%imdsTrain = imageDatastore(XTrain);
trainedNet = trainNetwork(XTrain,YTrain,NN_layers,options);


%% test neural network & visualization 
% Calculate accuracy

