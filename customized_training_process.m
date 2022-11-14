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


%% define network (dlnet)
NN_layers = [...
    ...
   ];

% convert to a layer graph
lgraph = layerGraph(NN_layers);
% Create a dlnetwork object from the layer graph.
dlnet = dlnetwork(lgraph);
% visualize the neural network
analyzeNetwork(dlnet)

%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :


%% Train neural network

% initialize the average gradients and squared average gradients
% averageGrad
% averageSqGrad

% "for-loop " for training

for epoch = 1:numEpochs
    
   % updae learnable parameters based on mini-batch of data
    for i = 1:numIterationsPerEpoch
        % Read mini-batch of data and convert the labels to dummy variables.


        % Convert mini-batch of data to a dlarray.
        
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        
        % Update the network parameters using the optimizer, like SGD, Adam
        
        % Calculate accuracy & show the training progress.

        % option: validation

    end
end


%% test neural network & visualization 

%% Define Model Gradients Function
% 
function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y)

    % forward propagation 
    dlYPred = forward(dlnet,dlX);
    % calculate loss -- varies based on different requirement
    loss = crossentropy(dlYPred,Y);
    % calculate gradients 
    gradients = dlgradient(loss,dlnet.Learnables);
    
end