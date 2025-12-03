function ypred = predict_knn_model(model, Xs)
% PREDICT_KNN_MODEL Predict labels for rows Xs using a model struct
% model can be either a struct with fields X (n x d), Y (n x 1), k
% or a trained ClassificationKNN object from Statistics Toolbox.
if isempty(Xs), ypred = []; return; end
if isa(model,'struct') && isfield(model,'X')
    Xtrain = double(model.X); ytrain = double(model.Y(:)); k = model.k;
    Xs = double(Xs);
    m = size(Xs,1); ypred = zeros(m,1);
    for i = 1:m
        dif = Xtrain - repmat(Xs(i,:), size(Xtrain,1), 1);
        D = sqrt(sum(dif.^2,2)); [~, idxs] = sort(D);
        kidx = idxs(1:min(k, length(idxs)));
        ypred(i) = mode(ytrain(kidx));
    end
elseif exist('ClassificationKNN','class') && isa(model,'ClassificationKNN')
    ypred = predict(model, Xs);
else
    % last resort: try generic predict
    try ypred = predict(model, Xs); catch, error('Unsupported model type for predict_knn_model'); end
end
end
