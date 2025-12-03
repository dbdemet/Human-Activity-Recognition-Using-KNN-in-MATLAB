function [X, y, subjects] = load_prepare(dataDir)
%LOAD_PREPARE Load UCI HAR precomputed features (X) and labels (y).
%   [X,y,subjects] = load_prepare(dataDir)
%   If dataDir is not provided, assumes 'UCI HAR Dataset' in cwd.

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(pwd,'UCI HAR Dataset');
end

trainXfile = fullfile(dataDir,'train','X_train.txt');
testXfile  = fullfile(dataDir,'test','X_test.txt');
try
    Xtrain = readmatrix(trainXfile);
    Xtest  = readmatrix(testXfile);
catch ME
    error('Files could not be read. Please check folder %s structure. Error: %s', dataDir, ME.message);
end

ytrain = readmatrix(fullfile(dataDir,'train','y_train.txt'));
ytest  = readmatrix(fullfile(dataDir,'test','y_test.txt'));
strain = readmatrix(fullfile(dataDir,'train','subject_train.txt'));
stest  = readmatrix(fullfile(dataDir,'test','subject_test.txt'));

X = [Xtrain; Xtest];
y = [ytrain; ytest];
subjects = [strain; stest];

end
