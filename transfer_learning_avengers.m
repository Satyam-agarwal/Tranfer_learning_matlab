

ds=imageDatastore('self','IncludeSubfolders',true,'LabelSource','foldernames');
actornames=ds.Labels
net=alexnet;

for i = 1:148
    data=readimage(ds,i);
    data=imresize(data,[227 227]);
    
    
    
end  

[ds1,ds2]= splitEachLabel(ds,0.6);
net.Layers
%mylayers(23) = fullyConnectedLayer(2)

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(ds1.Labels))
inputSize = net.Layers(1).InputSize;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),ds1, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),ds2);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(netTransfer,augimdsValidation);
idx = randperm(numel(ds2.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(ds2,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

