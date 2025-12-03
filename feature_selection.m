function [selIdx, selNames, reducedFeatures, pcaModel] = feature_selection(features, labels, method, opts)
%FEATURE_SELECTION Select or reduce features using several methods.
%   [selIdx, selNames, reducedFeatures, pcaModel] = feature_selection(features, labels, method, opts)
%   method: 'relieff' (default), 'pca', 'sequentialfs'
%   opts.numFeatures (for relieff/sequentialfs), opts.pcaVariance (for PCA)

if nargin < 3 || isempty(method), method = 'relieff'; end
if nargin < 4, opts = struct(); end

switch lower(method)
    case 'relieff'
        if ~isfield(opts,'numFeatures'), opts.numFeatures = min(50,size(features,2)); end
        fprintf('Running ReliefF to rank features (Statistics Toolbox)...\n');
        % Use relieff from Statistics and Machine Learning Toolbox
        if exist('relieff','file')==2
            [ranked,weights] = relieff(features, labels, 10);
            selIdx = ranked(1:opts.numFeatures);
            fprintf('  ReliefF completed using Statistics Toolbox. Selected top %d features.\n', opts.numFeatures);
        else
            fprintf('  Warning: relieff not available — using F-score (ANOVA-style) fallback.\n');
            % Compute F-score for each feature (between / within class variance)
            X = features; y = labels(:);
            classes = unique(y);
            N = size(X,1);
            k = numel(classes);
            Fscore = zeros(1,size(X,2));
            overall_mean = mean(X,1);
            for f = 1:size(X,2)
                sb = 0; sw = 0;
                for ci = 1:k
                    idx = (y==classes(ci));
                    ni = sum(idx);
                    if ni==0, continue; end
                    mean_i = mean(X(idx,f));
                    sb = sb + ni * (mean_i - overall_mean(f))^2; % between-class sum squares
                    sw = sw + sum( (X(idx,f) - mean_i).^2 ); % within-class sum squares
                end
                if sw == 0
                    Fscore(f) = 0;
                else
                    msb = sb / (k-1);
                    msw = sw / (N-k);
                    Fscore(f) = msb / (msw + eps);
                end
            end
            [~, ranked] = sort(Fscore, 'descend');
            selIdx = ranked(1:opts.numFeatures);
        end
        reducedFeatures = features(:, selIdx);
        selNames = arrayfun(@(i) sprintf('F%d', i), selIdx, 'UniformOutput', false);
        pcaModel = [];

    case 'pca'
        if ~isfield(opts,'pcaVariance'), opts.pcaVariance = 0.95; end
        fprintf('Running PCA to explain %.2f variance (Statistics Toolbox)...\n', opts.pcaVariance);
        % Use pca function from Statistics and Machine Learning Toolbox
        [coeff, score, latent, tsq, explained, mu] = pca(features);
        cum = cumsum(explained)/100;
        k = find(cum >= opts.pcaVariance, 1, 'first');
        reducedFeatures = score(:,1:k);
        selIdx = 1:k;
        selNames = arrayfun(@(i) sprintf('PC%d', i), selIdx, 'UniformOutput', false);
        pcaModel.coeff = coeff; pcaModel.mu = mu; pcaModel.explained = explained;

    case 'sequentialfs'
        if ~isfield(opts,'numFeatures'), opts.numFeatures = min(30,size(features,2)); end
        fprintf('Running sequential feature selection (forward) using Statistics Toolbox...\n');
        % Use sequentialfs, fitcensemble, and cvpartition from Statistics and Machine Learning Toolbox
        if exist('sequentialfs','file')==2 && exist('fitcensemble','file')==2 && exist('cvpartition','file')==2
            fun = @(trainX, trainY, testX, testY) sum(predict(fitcensemble(trainX, trainY,'Method','Bag'), testX) ~= testY);
            optsfs = statset('display','iter');
            try
                [sel, history] = sequentialfs(fun, features, labels, 'cv', cvpartition(labels,'KFold',5), 'NFeatures', opts.numFeatures, 'options', optsfs);
                selIdx = find(sel);
                reducedFeatures = features(:, selIdx);
                selNames = arrayfun(@(i) sprintf('F%d', i), selIdx, 'UniformOutput', false);
                pcaModel = [];
            catch ME
                fprintf('  sequentialfs failed: %s — falling back to F-score ranking.\n', ME.message);
                % fall back to relieff branch behavior
                method = 'relieff';
                % recurse to relieff handling
                [selIdx, selNames, reducedFeatures, pcaModel] = feature_selection(features, labels, method, opts);
                return;
            end
        else
            fprintf('  sequentialfs/fitcensemble/cvpartition unavailable — falling back to F-score ranking.\n');
            method = 'relieff';
            [selIdx, selNames, reducedFeatures, pcaModel] = feature_selection(features, labels, method, opts);
            return;
        end

    otherwise
        error('Unknown method: %s', method);
end

fprintf('Feature selection finished. Selected %d features.\n', size(reducedFeatures,2));
end
