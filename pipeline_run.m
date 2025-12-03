function pipeline_run(dataDir, opts)
%PIPELINE_RUN End-to-end pipeline: extract features, select, train, evaluate.
%   pipeline_run(dataDir, opts)
% opts.saveResults (default false) -> save pipeline_results.mat
% opts.summaryOnly (default true) -> only print accuracy/meanF1 and skip figures

if nargin < 1 || isempty(dataDir), dataDir = fullfile(pwd,'UCI HAR Dataset'); end
if nargin < 2, opts = struct(); end
if ~isfield(opts,'saveResults'), opts.saveResults = false; end
if ~isfield(opts,'summaryOnly'), opts.summaryOnly = true; end
if ~isfield(opts,'resultsDir'), opts.resultsDir = fullfile(pwd,'results_figs'); end
if ~isfield(opts,'overwrite'), opts.overwrite = false; end

% Ensure resultsDir does not overwrite existing runs unless overwrite=true
resultsDir = opts.resultsDir;
if exist(resultsDir,'dir')==7 && ~opts.overwrite
	ts = datestr(now,'yyyymmdd_HHMMSS');
	resultsDir = fullfile(resultsDir, ['run_' ts]);
	if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
else
	if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
end

fprintf('Extracting features...\n');
[F, L] = extract_features(dataDir);
fprintf('Running EDA...\n');
eda(dataDir, resultsDir);

fprintf('Selecting features (ReliefF)...\n');
[selIdx, selNames, Fsel] = feature_selection(F, L, 'relieff', struct('numFeatures',50));

fprintf('Training models on selected features...\n');
results = train_models(Fsel, L);

fprintf('Evaluating models...\n');
evaluate_models(results, Fsel, L, struct('summaryOnly', opts.summaryOnly, 'outDir', resultsDir));

if opts.saveResults
	save(fullfile(resultsDir,'pipeline_results.mat'),'results','selIdx','selNames','-v7.3');
	fprintf('Pipeline finished. Results saved to %s and figures in %s.\n', fullfile(resultsDir,'pipeline_results.mat'), resultsDir);
else
	fprintf('Pipeline finished. Figures (if any) are in %s. Results not saved (opts.saveResults=false).\n', resultsDir);
end
end
