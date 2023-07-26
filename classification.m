%% READING AND SPLITTING CLASSIFICATION DATASET

% Load images from folder and assign labels
imds = imageDatastore('17flowers', 'LabelSource', 'none');
numImages = numel(imds.Files);
labels = repelem(1:ceil(numImages/80), 80);
labels(numImages+1:end) = [];
imds.Labels = categorical(labels);

% Resize images to 256-by-256-by-3 pixels
imds.ReadFcn = @(filename) imresize(imread(filename), [256, 256]);

% Define data augmentation options
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-80 80], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXShear',[-20 20], ...
    'RandYShear',[-20 20], ...
    'RandScale',[1.2,1.5]);


% Split dataset into training, validation and test sets
rng(123); %random seed to ensure the split stays same
[imdsTrainVal, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrainVal, 0.8, 'randomized');

% Create augmented image datastore
augimds = augmentedImageDatastore([256 256 3], imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% CREATING AND TRAINING CNN Architecture

% Define CNN architecture
numClasses = 17;
layers = [imageInputLayer([256 256 3], 'Normalization','rescale-zero-one')

          convolution2dLayer(3,32,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(3,64,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(3,128,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer(3,192,'Padding','same')
          batchNormalizationLayer
          reluLayer

          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(3,256,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          convolution2dLayer(3,512,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(3,1024,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          maxPooling2dLayer(2,'Stride',2)
          
          convolution2dLayer(3,2048,'Padding','same')
          batchNormalizationLayer
          reluLayer
          
          globalAveragePooling2dLayer
          
          fullyConnectedLayer(1024)
          reluLayer
          dropoutLayer(0.5)

          fullyConnectedLayer(256)
          reluLayer
          dropoutLayer(0.5)

          fullyConnectedLayer(128)
          reluLayer
          dropoutLayer(0.5)
    
          fullyConnectedLayer(numClasses)
          softmaxLayer
          classificationLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 27, ...
    'ValidationPatience', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment', 'gpu');

% Train CNN
net = trainNetwork(augimds, layers, options);

%% SAVING THE MODEL
save('classnet.mat', 'net');

%% TRANSFER LEARNING MODELS

% Load pre-trained ResNet-18
numClasses = 17;
net = inceptionresnetv2; %replace this with whatever model you want to use

layers = [imageInputLayer([256 256 3],'Normalization','rescale-zero-one')
        net(2:end-3)

        fullyConnectedLayer(1000)
        reluLayer
        dropoutLayer(0.5)

        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

% Set up training options
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 6, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment', 'cpu');

% Train CNN
net = trainNetwork(imdsTrain, layers, options);

%% Calculating evaluation on test set

%load the trained model
load('classnet.mat','net')


% Classify test images using the loaded CNN model
predLabels = classify(net,imdsTest);

% Calculate evaluation measures
testLabels = imdsTest.Labels;
accuracy = mean(predLabels == testLabels);

% Print evaluation measures
fprintf('Accuracy: %0.2f%%\n', accuracy * 100);

%% Printing predicted class while displaying the image next to it for an image

% Test CNN on a sample image
load('classnet.mat','net')
%im = imread('17flowers/image_0001.jpg');
im = read(imdsTest);
%im = imresize(im, [256, 256]);
pred = classify(net, im);
imshow(im)
title(string(pred))

%% To visualise the model built
analyzeNetwork(net)