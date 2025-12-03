function results = train_models(features, labels, opts)
% TRAIN_MODELS Clean, toolbox-free trainer with K-fold CV
% Usage: results = train_models(features, labels, opts)

if nargin < 2, error('features and labels required'); end
if nargin < 3, opts = struct(); end
if ~isfield(opts,'K'), opts.K = 5; end
if ~isfield(opts,'tuneK'), opts.tuneK = false; end
if ~isfield(opts,'featureSelect'), opts.featureSelect = false; end
if ~isfield(opts,'includeCentroid'), opts.includeCentroid = true; end
if ~isfield(opts,'display'), opts.display = true; end

features = double(features); labels = double(labels(:)); N = size(features,1); K = opts.K;

% Build folds (try toolbox first)
useCv = false; folds = cell(K,1);
if exist('cvpartition','file')==2
    try cvp = cvpartition(labels,'KFold',K); useCv = true; catch, useCv = false; end
end
if ~useCv
    classes = unique(labels);
    for ci=1:numel(classes)
        idxs = find(labels==classes(ci)); idxs = idxs(randperm(numel(idxs)));
        for t=1:numel(idxs), f = mod(t-1,K)+1; folds{f}(end+1)=idxs(t); end
    end
end

defaultK = 5; modelNames = {'kNN_manual'}; if opts.includeCentroid, modelNames{end+1}='NearestCentroid'; end
M = numel(modelNames); preds_total = cell(M,1); for i=1:M, preds_total{i}=zeros(N,1); end

for fold=1:K
    if useCv, trainIdx = training(cvp,fold); testIdx = test(cvp,fold); else, testIdx=false(N,1); testIdx(folds{fold})=true; trainIdx=~testIdx; end
    Xtr = features(trainIdx,:); ytr = labels(trainIdx); Xte = features(testIdx,:);
    if isempty(Xte), continue; end

    sel = 1:size(features,2);
    if opts.featureSelect
        try sel = feature_selection(Xtr,ytr,'relieff',struct('nFeatures',min(50,size(Xtr,2)))); catch, sel = 1:size(features,2); end
    end

    bestK = defaultK;
    if isfield(opts,'bestK') && ~isempty(opts.bestK)
        bestK = opts.bestK;
    elseif opts.tuneK
        try [bestK,~] = tune_k_values(Xtr(:,sel), ytr); catch, bestK = defaultK; end
    end

    % k-NN manual prediction
    km = struct('X',double(Xtr(:,sel)),'Y',double(ytr),'k',bestK);
    preds_total{1}(testIdx) = predict_knn_model(km, double(Xte(:,sel)));

    % nearest centroid (recomputed per fold)
    if opts.includeCentroid
        classesTr = unique(ytr); cent = zeros(numel(classesTr), numel(sel));
        for ci=1:numel(classesTr), cent(ci,:)=mean(Xtr(ytr==classesTr(ci),sel),1); end
        ypred = zeros(size(Xte,1),1);
        for ii=1:size(Xte,1)
            dif = cent - repmat(Xte(ii,sel), size(cent,1),1);
            [~, idx] = min(sqrt(sum(dif.^2,2)));
            ypred(ii) = classesTr(idx);
        end
        preds_total{2}(testIdx) = ypred;
    end
end

results = struct(); results.names = modelNames; results.metrics = cell(M,1); results.opts = opts;
for mi = 1:M
    preds = preds_total{mi};
    if exist('confusionmat','file')==2
        C = confusionmat(labels, preds);
    else
        classesU = unique([labels(:); preds(:)]); nC = numel(classesU); C=zeros(nC,nC);
        for ii=1:N, t=find(classesU==labels(ii)); p=find(classesU==preds(ii)); if ~isempty(t)&&~isempty(p), C(t,p)=C(t,p)+1; end; end
    end
    acc = sum(diag(C))/sum(C(:)); pvec=zeros(size(C,1),1); rvec=zeros(size(C,1),1); f1=zeros(size(C,1),1);
    for c=1:size(C,1), tp=C(c,c); fp=sum(C(:,c))-tp; fn=sum(C(c,:))-tp; if tp+fp>0, pvec(c)=tp/(tp+fp); end; if tp+fn>0, rvec(c)=tp/(tp+fn); end; if pvec(c)+rvec(c)>0, f1(c)=2*pvec(c)*rvec(c)/(pvec(c)+rvec(c)); end; end
    metrics.confusion=C; metrics.accuracy=acc; metrics.precision=pvec; metrics.recall=rvec; metrics.f1=f1; metrics.preds=preds; results.metrics{mi}=metrics;
end

if isfield(opts,'display') && opts.display
    for i=1:numel(results.names), fprintf('%s: Accuracy=%.2f%%, MeanF1=%.3f\n', results.names{i}, results.metrics{i}.accuracy*100, mean(results.metrics{i}.f1)); end
    % Note: Detailed confusion matrix visualization is handled by evaluate_models.m
    % This is just a quick preview
    try
        figure('Name','train_models - confusion preview');
        imagesc(results.metrics{1}.confusion); 
        colormap('bone'); colorbar; 
        xlabel('Predicted'); ylabel('True');
        title(sprintf('%s - Confusion matrix (preview)', results.names{1}));
    catch
    end
end

end

