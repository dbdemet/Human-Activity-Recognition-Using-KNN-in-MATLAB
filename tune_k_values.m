function [bestK, kStats] = tune_k_values(X, y, kRange, cvFolds)
% TUNE_K_VALUES Find best k for k-NN by cross-validation (toolbox-free)
%   [bestK, kStats] = tune_k_values(X, y) tries odd ks 1:2:15 with 5-fold CV.
%   [bestK, kStats] = tune_k_values(X, y, kRange, cvFolds) specifies range and folds.

if nargin < 3 || isempty(kRange), kRange = 1:2:15; end
if nargin < 4 || isempty(cvFolds), cvFolds = 5; end
X = double(X); y = double(y(:));
N = size(X,1);

% Ensure kRange are positive integers
kRange = unique(max(1, round(kRange(:))));

% Prepare CV partition (prefer toolbox `cvpartition`, otherwise build a stratified fallback)
try
    cvp = cvpartition(y,'KFold',cvFolds);
catch
    % fallback: build a stratified fold assignment vector `foldIdx` of length N
    cvp = [];
    % sanitize cvFolds relative to N
    cvFolds = max(2, min(cvFolds, N));
    % stratified distribution: for each class, shuffle indices and assign round-robin
    foldIdx = zeros(N,1);
    rngState = []; % do not change user's RNG persistently
    try
        rngState = rng();
    catch
        rngState = [];
    end
    for cls = unique(y)'
        ids = find(y == cls);
        if isempty(ids), continue; end
        % shuffle class indices
        rp = ids(randperm(length(ids)));
        for ii = 1:length(rp)
            f = mod(ii-1, cvFolds) + 1;
            foldIdx(rp(ii)) = f;
        end
    end
    % If any samples left unassigned (shouldn't happen), assign them round-robin
    un = find(foldIdx==0);
    for ii = 1:length(un)
        f = mod(ii-1, cvFolds) + 1;
        foldIdx(un(ii)) = f;
    end
    % restore RNG state if we changed it
    if ~isempty(rngState)
        try rng(rngState); catch, end
    end
end

kStats = struct('k', num2cell(kRange),'meanAcc',[],'stdAcc',[]);

for ki = 1:length(kRange)
    k = kRange(ki);
    accs = zeros(cvFolds,1);
    for fold = 1:cvFolds
        if ~isempty(cvp)
            trainIdx = training(cvp,fold); testIdx = test(cvp,fold);
        else
            % use stratified foldIdx built above
            testIdx = (foldIdx == fold);
            trainIdx = ~testIdx;
        end
        Xtr = X(trainIdx,:); ytr = y(trainIdx);
        Xte = X(testIdx,:); yte = y(testIdx);
        if isempty(Xte)
            accs(fold) = NaN; continue;
        end
        % use predict_knn_model helper if available
        try
            model = struct('X', Xtr, 'Y', ytr, 'k', k);
            ypred = predict_knn_model(model, Xte);
        catch
            % fallback: manual distance loop
            m = size(Xte,1); ypred = zeros(m,1);
            for ii = 1:m
                dif = Xtr - repmat(Xte(ii,:), size(Xtr,1), 1);
                D = sqrt(sum(dif.^2,2)); [~, idxs] = sort(D);
                kidx = idxs(1:min(k, length(idxs)));
                ypred(ii) = mode(ytr(kidx));
            end
        end
        accs(fold) = mean(ypred == yte);
    end
    valid = ~isnan(accs);
    if any(valid)
        kStats(ki).meanAcc = mean(accs(valid));
        kStats(ki).stdAcc = std(accs(valid));
    else
        kStats(ki).meanAcc = NaN;
        kStats(ki).stdAcc = NaN;
    end
end

[~, bestIdx] = max([kStats.meanAcc]);
bestK = kStats(bestIdx).k;
fprintf('Best k = %d (mean accuracy = %.3f)\n', bestK, kStats(bestIdx).meanAcc);
end
