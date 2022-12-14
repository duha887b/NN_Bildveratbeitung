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
% lade Trainingsdatensatz
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

XImages = X1;
YLabels = Y1;

% Aufteilen der Daten in Test, Validierung und Trainingsdaten
[trainInd,testInd,validInd] = dividerand(numel(YLabels),0.7,0.1,0.2);

XTrain = XImages(:,:,:,trainInd);
YTrain = YLabels(trainInd);
XTest = XImages(:,:,:,testInd);
YTest = YLabels(testInd);
XValid = XImages(:,:,:,validInd);
YValid = YLabels(validInd);

%% define network (dlnet)

inputSize = [28 28 1];
layer0 = imageInputLayer(inputSize,Normalization="none" );

outputSize = 392;
layer1 = fullyConnectedLayer(outputSize);

layer2 = reluLayer;

layer3 = fullyConnectedLayer(10);

layer4 = softmaxLayer;


NN_layers = [ ...
    layer0
    layer1
    layer2
    layer3
    layer4
    ];


% convert to a layer graph
lgraph = layerGraph(NN_layers);
% Create a dlnetwork object from the layer graph.
dlnet = dlnetwork(lgraph);
% visualize the neural network
analyzeNetwork(dlnet)

%% Specify Training Options (define hyperparameters)

miniBatchSize = 128;

numEpochs = 30;

learnRate = 0.01;

executionEnvironment = 'auto';

numIterationsPerEpoch = floor(numel(YTrain)/miniBatchSize);

maxIteration = numEpochs * numIterationsPerEpoch;

classes = categories(YTrain);

updateMonitorIter = 10;

%% Train neural network

% initialize the average gradients and squared average gradients
averageGrad = [];
averageSqGrad = [];

% ini trainig progress monitor
monitor = trainingProgressMonitor;
monitor.Info = ["Learning_rate","Epoch","Iteration","Training_Accuracy","Validation_Accuracy"];
monitor.Metrics = ["TrainingLoss","ValidationLoss","TrainingAccuracy","ValidationAccuracy"];

monitor.XLabel = "Iteration";
groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);
groupSubPlot(monitor,"Accuracy(%)",["TrainingAccuracy","ValidationAccuracy"]);

iterations = 0;

% "for-loop " for training
monitor.Status = "Runnig";
for epoch = 1:numEpochs
    
   
    if monitor.Stop
        break
    end

    % updae learnable parameters based on mini-batch of data

    count=5;
    for i = 1:numIterationsPerEpoch

         if monitor.Stop
            break
         end

        iterations = iterations +1;

        % Read mini-batch of data and convert the labels to dummy variables.
        
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XTmp = XTrain(:,:,:,idx);
        XTmp = single(XTmp);

        Y = zeros(10, miniBatchSize,"single");
        for c = 1:10
            Y(c,YTrain(idx)==classes(c)) = 1;
        end
        


        % Convert mini-batch of data to a dlarray
        dlX = dlarray(XTmp,'SSCB');
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        [gradients,loss,dlYPred] = dlfeval(@modelGradients,dlnet,dlX,Y);
        
        % Update the network parameters using the optimizer, like SGD, Adam
        %1 ADAM
        [dlnet,averageGrad,averageSqGrad ] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iterations,learnRate);
        
        % Calculate accuracy & show the training progress.
        if count < i
            % validation accuracy
            YVPred = predict(dlnet,dlarray(single(XValid),'SSCB'));
            [~,idx] = max(extractdata(YVPred),[],1);
            YVPred = classes(idx);
            ValAccuracy = mean(YVPred==YValid);
            % training accuracy
            accuracy = 1-loss;
            
            fprintf("Epoche: %d ; Iteration: %d ; Accuracy: %d  ; Validation: %d\n ",epoch,iterations,accuracy,ValAccuracy);
            
            % update training Monitor (Aufgabe 1)

            updateInfo(monitor,Learning_rate=learnRate,Epoch= string(epoch) + " of " + string(numEpochs), ...
                        Iteration = string(iterations) + " of " + string(maxIteration), ...
                        Training_Accuracy= string((1-loss)*100) +"%", ...
                        Validation_Accuracy= string((ValAccuracy)*100)+"%");
            
            recordMetrics(monitor,iterations, ...
                           TrainingLoss=loss, ...
                           TrainingAccuracy=(1-loss)*100, ...
                           ValidationLoss=(1-ValAccuracy), ...
                           ValidationAccuracy=ValAccuracy*100);


            monitor.Progress = 100*iterations/maxIteration;
            count = count + updateMonitorIter;
        end

        

        

    end
end

monitor.Status = "Done";

%% test neural network & visualization (Diagramme Aufgabe 2)

% Calculate accuracy

YPred = predict(dlnet,dlarray(single(XTest),'SSCB'));
[~,idx] = max(extractdata(YPred),[],1);
YPred = classes(idx);
accuracy = mean(YPred==YTest);

%daten in wichtungsMatrix eintragen 
weights = zeros(11,11);
for i =1:numel(YTest)
    
    weights(double(YTest(i))+1,str2num(YPred{i})+1) = weights(double(YTest(i))+1,str2num(YPred{i})+1) +1;
    
end 

for kc = 1:10                                                       % aus Matrix genauigkeit der ziffern berechnen 
    tmp = 0;
    tmp2 = 0;
    for kr = 2:11
        if kc == kr-1 
            tmp = weights(kr,kc);                                   % anzahl der richtig vorhergesagten Ziffern
        end

        tmp2 = tmp2 + weights(kr,kc);                               % Aufsummierung der gesamten Versuche pro ziffer
        
    end
    scatter(kc-1,tmp/tmp2,100,'x','MarkerEdgeColor',[0 0 1]);       % eintragen in Diagramm
    
    grid on
    hold on
    
end
scatter(1:8,accuracy,15000,"red","_");
scatter(0,accuracy,10,"red","_");
scatter(9,accuracy,10,"red","_");

title('??=f(Ziffer)')
xlabel('Ziffer')
ylabel('Accuracy ??')

hold off

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