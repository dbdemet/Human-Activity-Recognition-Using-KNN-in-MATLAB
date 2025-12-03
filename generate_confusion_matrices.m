function generate_confusion_matrices(dataDir, opts)
%GENERATE_CONFUSION_MATRICES Generate and save confusion matrices for all models
%   generate_confusion_matrices(dataDir, opts)
%   This script loads results, generates confusion matrices with activity labels,
%   and saves them to results_figs folder.
%
%   Example:
%     generate_confusion_matrices('UCI HAR Dataset');
%     generate_confusion_matrices('UCI HAR Dataset', struct('loadFromFile', true));

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(pwd, 'UCI HAR Dataset');
end
if nargin < 2, opts = struct(); end
if ~isfield(opts, 'loadFromFile'), opts.loadFromFile = false; end
if ~isfield(opts, 'outDir'), opts.outDir = fullfile(pwd, 'results_figs'); end

% Ensure output directory exists
if ~exist(opts.outDir, 'dir'), mkdir(opts.outDir); end

fprintf('=== Generating Confusion Matrices ===\n');

% Load results
if opts.loadFromFile && exist('har_models_results.mat', 'file')
    fprintf('Loading results from har_models_results.mat...\n');
    load('har_models_results.mat', 'results', 'features', 'labels');
    if ~exist('features', 'var') || ~exist('labels', 'var')
        fprintf('Warning: features or labels not found in file. Extracting from dataset...\n');
        [features, labels] = extract_features(dataDir, 'train');
        [Ftest, Ltest] = extract_features(dataDir, 'test');
        features = [features; Ftest];
        labels = [labels; Ltest];
    end
else
    fprintf('Extracting features from dataset...\n');
    [features, labels] = extract_features(dataDir, 'train');
    [Ftest, Ltest] = extract_features(dataDir, 'test');
    features = [features; Ftest];
    labels = [labels; Ltest];
    
    fprintf('Running feature selection...\n');
    [selIdx, ~, features] = feature_selection(features, labels, 'relieff', struct('numFeatures', 50));
    
    fprintf('Training models...\n');
    results = train_models(features, labels, struct('K', 5, 'tuneK', false, 'bestK', 5));
end

fprintf('Generating confusion matrices...\n');
% Use evaluate_models to generate confusion matrices
evaluate_models(results, features, labels, struct('summaryOnly', false, 'outDir', opts.outDir, 'overwrite', true));

fprintf('\n=== Confusion Matrices Generated Successfully ===\n');
fprintf('Saved to: %s\n', opts.outDir);

end



