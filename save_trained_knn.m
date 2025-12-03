function save_trained_knn(model, outFile)
% SAVE_TRAINED_KNN Save a simple model struct to disk (mat file)
%   save_trained_knn(model) saves to 'knn_model.mat' in cwd.
if nargin < 2 || isempty(outFile), outFile = fullfile(pwd,'knn_model.mat'); end
if isstruct(model)
    save(outFile,'model','-v7.3');
else
    % try to extract fields for portability
    try
        mstruct.X = model.X; mstruct.Y = model.Y; mstruct.k = model.k;
        save(outFile,'mstruct','-v7.3');
    catch
        save(outFile,'model','-v7.3');
    end
end
fprintf('Saved model to %s\n', outFile);
end
