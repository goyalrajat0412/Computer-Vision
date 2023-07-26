%% OTSU segmentation: 

% Load input image
I = imread('17flowers/image_0001.jpg');
Ig = rgb2gray(I);
imshow(Ig)
level = multithresh(Ig);
seg_Ig = imquantize(Ig,level);
figure
imshow(seg_Ig,[]);
I_masked = I .* repmat( cast(seg_Ig-1, class(I)), 1, 1, size(I,3) );
figure
imshow(I_masked);

%% DIRECTORIES TO SPLIT DATASET INTO TRAIN, VAL & TEST

% Set the paths to the source and destination folders
img_src_folder = 'daffodilSeg/ImagesRsz256/';
lab_src_folder = 'daffodilSeg/LabelsRsz256/';

%% TO MOVE TEST DATA INTO DESIRED FOLDER

imgTest_dst_folder = 'daffodilSeg/ImagesRsz256Test/';
labTest_dst_folder = 'daffodilSeg/LabelsRsz256Test/';

% Get a list of all image files in the source folder
img_files = dir(fullfile(img_src_folder, '*.png'));

% Get the number of images in the folder
num_images = numel(img_files);

% Calculate the number of images to move
num_to_move = round(num_images * 0.10);

% Randomly select the indices of the images to move
indices = randperm(num_images, num_to_move);

% Move the Image test images to the destination folder
for i = 1:numel(indices)
    % Get the filename of the image to move
    filename = img_files(indices(i)).name;
    
    % Construct the full path to the source and destination files
    src_file = fullfile(img_src_folder, filename);
    dst_file = fullfile(imgTest_dst_folder, filename);
    
    % Move the file to the destination folder
    movefile(src_file, dst_file);
end


% Move the Label test images to the destination folder
for i = 1:numel(indices)
    % Get the filename of the image to move
    filename = img_files(indices(i)).name;
    
    % Construct the full path to the source and destination files
    src_file = fullfile(lab_src_folder, filename);
    dst_file = fullfile(labTest_dst_folder, filename);
    
    % Move the file to the destination folder
    movefile(src_file, dst_file);
end

%% TO MOVE VAL SET INTO DESIRED FOLDER

imgVal_dst_folder = 'daffodilSeg/ImagesRsz256Val/';
labVal_dst_folder = 'daffodilSeg/LabelsRsz256Val/';


% Get a list of all image files in the source folder
img_files = dir(fullfile(img_src_folder, '*.png'));

% Get the number of images in the folder
num_images = numel(img_files);

% Randomly select the indices of the images to move
indices = randperm(num_images, num_to_move);

% Move the Image val images to the destination folder
for i = 1:numel(indices)
    % Get the filename of the image to move
    filename = img_files(indices(i)).name;
    
    % Construct the full path to the source and destination files
    src_file = fullfile(img_src_folder, filename);
    dst_file = fullfile(imgVal_dst_folder, filename);
    
    % Move the file to the destination folder
    movefile(src_file, dst_file);
end


% Move the Label val images to the destination folder
for i = 1:numel(indices)
    % Get the filename of the image to move
    filename = img_files(indices(i)).name;
    
    % Construct the full path to the source and destination files
    src_file = fullfile(lab_src_folder, filename);
    dst_file = fullfile(labVal_dst_folder, filename);
    
    % Move the file to the destination folder
    movefile(src_file, dst_file);
end

%% READING AND SPLITTING SEGMENTATION DATASET

rng(123);
imageDirTrain = fullfile('daffodilSeg','ImagesRsz256');
labelDirTrain = fullfile('daffodilSeg','LabelsRsz256');

imageDirTest = fullfile('daffodilSeg','ImagesRsz256Test');
labelDirTest = fullfile('daffodilSeg','LabelsRsz256Test');

imageDirVal = fullfile('daffodilSeg','ImagesRsz256Val');
labelDirVal = fullfile('daffodilSeg','LabelsRsz256Val');

imdsTrain = imageDatastore(imageDirTrain); 
imdsTest = imageDatastore(imageDirTest); 
imdsVal = imageDatastore(imageDirVal);

numClasses = 2;

classNames = ["flower","background"]; % our class labels
labelIDs   = [1 3];

pxdsTrain = pixelLabelDatastore(labelDirTrain,classNames,labelIDs);
pxdsTest = pixelLabelDatastore(labelDirTest,classNames,labelIDs);
pxdsVal = pixelLabelDatastore(labelDirVal,classNames,labelIDs);

trainingData = pixelLabelImageDatastore(imdsTrain,pxdsTrain); %train set (images and pixel labels combined)
validationData = pixelLabelImageDatastore(imdsVal,pxdsVal); %validation set

%%

[imdsTrainVal, imdsTestVal] = splitEachLabel(imdsTrain, 0.8, 'randomized');

%% CALCULATING CLASS WEIGHTS

tbl = countEachLabel(pxdsTrain)  % Table of frequencies of pixels belonging to each class

%calculate class weights
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency

%% CREATING AND TRAINING CNN ARCHITECTURE

%Self-made architecture for segmentation
layers = [
    % Downsampling layers
    imageInputLayer([256 256 3], 'Normalization','zerocenter')
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
   
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    %Upsampling layers
    transposedConv2dLayer(2, 256, 'Stride', 2, 'Cropping', 'same') %heavy
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()

    transposedConv2dLayer(2, 128, 'Stride', 2, 'Cropping', 'same') %light
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()

    transposedConv2dLayer(2, 64, 'Stride', 2, 'Cropping', 'same') %lighter
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    
    transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same')
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    
    transposedConv2dLayer(2, 16, 'Stride', 2, 'Cropping', 'same')
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer()
    reluLayer()
    
    convolution2dLayer(1, numClasses)
    softmaxLayer()
    pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',classWeights);
    ]


options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',500, ...
    'MiniBatchSize', 19, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 3, ...
    'ValidationPatience', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment', 'gpu')
    
net = trainNetwork(trainingData, layers, options)

%% SAVING THE MODEL
save('segmentnet.mat', 'net') 

%% TRANSFER LEARNING MODEL

imageSize = [256 256 3];

% Create U-Net model using unetLayers
unetModel = unetLayers(imageSize, numClasses);
%analyzeNetwork(unetModel);
layers = unetModel.Layers;
lgraph = layerGraph(layers);
pixelLayer = [ pixelClassificationLayer('Classes', tbl.Name, 'ClassWeights', classWeights) ];
lgraph = replaceLayer(lgraph,'Segmentation-Layer',pixelLayer); %replace pixel classification layer of unet to our own pixel classification layer as unet doesnt use class weights in its final layer

%breaking and replacing the final layer in unet resulted in detaching of
%decoder with some upsampling layers so we fixed that
lgraph = connectLayers(lgraph, 'Decoder-Stage-1-UpReLU' ,'Decoder-Stage-1-DepthConcatenation/in2');
lgraph = connectLayers(lgraph, 'Decoder-Stage-2-UpReLU' ,'Decoder-Stage-2-DepthConcatenation/in2');
lgraph = connectLayers(lgraph, 'Decoder-Stage-3-UpReLU' ,'Decoder-Stage-3-DepthConcatenation/in2');
lgraph = connectLayers(lgraph, 'Decoder-Stage-4-UpReLU' ,'Decoder-Stage-4-DepthConcatenation/in2');

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',500, ...
    'MiniBatchSize', 19, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 3, ...
    'ValidationPatience', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment', 'gpu')
    
net = trainNetwork(trainingData, lgraph, options)

%% CALCULATING EVALUATION METRICS ON TEST SET

% Load the segmentation model
load('segmentnet.mat','net')

% Segment the test images using the model
predictedLabels = semanticseg(imdsTest, net);

% Evaluate the predicted labels against the ground truth labels
metrics = evaluateSemanticSegmentation(predictedLabels, pxdsTest);

% Display the evaluation metrics
disp(metrics);

%% DISPLAY A SEGMENTED IMAGE ALONGSIDE ITS GROUNDTRUTH MAP

load('segmentnet.mat','net')
I = read(imdsTest); %image
C = read(pxdsTest); %groundtruth map
I = imresize(I,5);
L = imresize(uint8(C{1}),5);

segmentationTest = semanticseg(I,net); %segmentedimage

%display overlaid
segmentationTestOverlay = labeloverlay(I,segmentationTest);

imshowpair(segmentationTestOverlay,L,'montage')

%% TO VISUALISE THE MODEL
analyzeNetwork(net);