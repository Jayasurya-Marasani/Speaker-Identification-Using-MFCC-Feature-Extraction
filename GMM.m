audioFolderPath = 'Add the path of the audio files folder';
audioFiles = dir(fullfile(audioFolderPath, '*.flac'));
data = cell(numel(audioFiles), 2);
for i = 1:numel(audioFiles)
    file = audioFiles(i);
    match = regexp(file.name, '(\d+)-', 'tokens');
    speakerID = str2double(match{1}{1});
    filePath = fullfile(audioFolderPath, file.name);
    data{i, 1} = filePath;
    data{i, 2} = speakerID;
end
dataTable = cell2table(data, 'VariableNames', {'AudioFile', 'SpeakerID'});
%% Feature extraction using MFCC
numFiles = size(dataTable, 1);
for i = 1:numFiles
    filePath = dataTable.AudioFile{i};
    speakerID = dataTable.SpeakerID(i);
    
    % Read the audio file (FLAC to WAV conversion if necessary)
    [audio, sampleRate] = audioread(filePath);
    
    % Perform MFCC feature extraction
    mfccFeatures = mfcc(audio, sampleRate);
    
    % Store the MFCC features in your data table or any other desired structure
    % e.g., add a new column to the data table
    dataTable.MFCCFeatures{i} = mfccFeatures;
end
%%
trainRatio = 0.8;
seed = 42;
[trainData, testData] = splitData(dataTable, trainRatio, seed);
%%
% Create a GMM model for each speaker ID
uniqueSpeakers = unique(trainData.SpeakerID);
numSpeakers = numel(uniqueSpeakers);
gmmModels = cell(numSpeakers, 1);
numComponents = numSpeakers;
for i = 1:numSpeakers
    speakerID = uniqueSpeakers(i);
    
    % Get the MFCC features for the current speaker
    speakerFeatures = trainData.MFCCFeatures(trainData.SpeakerID == speakerID);
    
    % Find the maximum number of frames among the speaker's MFCC features
    maxFrames = max(cellfun(@(x) size(x, 2), speakerFeatures));
    
    % Pad or truncate the MFCC features to have the same number of frames
    paddedFeatures = cellfun(@(x) [x, zeros(size(x, 1), maxFrames - size(x, 2))], speakerFeatures, 'UniformOutput', false);
    
    % Concatenate all the speaker's MFCC features into a single matrix
    speakerFeaturesMatrix = cat(1, paddedFeatures{:});
    
    % Train a GMM model for the current speaker
    gmmModels{i} = fitgmdist(speakerFeaturesMatrix, 1,'Options', statset('MaxIter', 200));
end

%%
% Initialize variables for accuracy calculation
numTestFiles = size(testData, 1);
predictedSpeakerIDs = zeros(numTestFiles, 1);
correctSpeakerIDs = testData.SpeakerID;

% Perform speaker identification on test data
for i = 1:numTestFiles
    % Get the MFCC features for the current test file
    testFeatures = testData.MFCCFeatures{i};
    
    % Calculate the log-likelihood scores for each GMM model
    scores = zeros(numSpeakers, 1);
    for j = 1:numSpeakers
        score = log(pdf(gmmModels{j}, testFeatures));
        scores(j) = sum(score);
    end
    
    % Identify the predicted speaker ID based on the highest score
    [~, predictedSpeakerIdx] = max(scores);
    predictedSpeakerIDs(i) = uniqueSpeakers(predictedSpeakerIdx);
end

% Calculate accuracy
accuracy = sum(predictedSpeakerIDs == correctSpeakerIDs) / numTestFiles;
disp(accuracy*100);
%%
% Calculate the confusion matrix
C = confusionmat(correctSpeakerIDs, predictedSpeakerIDs);
% Define a custom colormap
custom_colormap = [1, 1, 1; flipud(parula(64))];
custom_colormap = custom_colormap(1:end-10, :); % Exclude the brightest colors

% Plot the confusion matrix with the custom colormap
figure;
heatmap(uniqueSpeakers, uniqueSpeakers, C, 'Colormap', custom_colormap, 'ColorbarVisible', 'off');
xlabel('Predicted Label');
ylabel('True Label');
title('Confusion Matrix');
%%
% Calculate the confusion matrix
cm = confusionmat(correctSpeakerIDs, predictedSpeakerIDs);
% Calculate precision, recall, and F1-score
numClasses = numel(uniqueSpeakers);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    truePositives = cm(i, i);
    falsePositives = sum(cm(:, i)) - truePositives;
    falseNegatives = sum(cm(i, :)) - truePositives;
    
    precision(i) = truePositives / (truePositives + falsePositives);
    recall(i) = truePositives / (truePositives + falseNegatives);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Print the classification report
fprintf('\nClassification Report:\n');
fprintf('Class\t\tPrecision\tRecall\t\tF1-Score\n');
for i = 1:numClasses
    fprintf('%d\t\t%.4f\t\t%.4f\t\t%.4f\n', uniqueSpeakers(i), precision(i), recall(i), f1Score(i));
end