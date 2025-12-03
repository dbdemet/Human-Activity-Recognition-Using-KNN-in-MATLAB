function varargout = show_results(varargin)
%SHOW_RESULTS Unified display/training helper (moved from show_results_summary)
% Usage:
%  show_results(F, L)                - train with defaults and display
%  show_results(F, L, doTune)        - doTune true/false (legacy call)
%  show_results(results)             - display a results struct (no training)
%  show_results(results, labels)     - display per-class metrics using labels
%  show_results(..., opts)           - optional struct to control display:
%       opts.saveFigures (default false)
%       opts.showAllConfusions (default false)
%       opts.summaryOnly (default false)

% Forward to the implementation that used to live in show_results_summary.m
% Accept variable input/output and behave like the previous unified helper.
% Extract optional `opts` (4th vararg) safely from varargin before use.
opts = struct();
if numel(varargin) >= 4 && ~isempty(varargin{4})
	opts = varargin{4};
end
if ~isfield(opts,'saveFigures'), opts.saveFigures = false; end
if ~isfield(opts,'showAllConfusions'), opts.showAllConfusions = false; end
if ~isfield(opts,'summaryOnly'), opts.summaryOnly = false; end
if ~isfield(opts,'showPerClass'), opts.showPerClass = []; end

arg1 = [];
if ~isempty(varargin), arg1 = varargin{1}; end
arg2 = []; if numel(varargin) >= 2, arg2 = varargin{2}; end
arg3 = []; if numel(varargin) >= 3, arg3 = varargin{3}; end

% Determine mode: display-only if first arg is a results struct
isResultsStruct = isstruct(arg1) && isfield(arg1,'names') && isfield(arg1,'metrics');
if isResultsStruct
	results = arg1; labels = [];
	if ~isempty(arg2), labels = arg2; end
else
	% training mode: support legacy signature show_results(F, L, doTune)
	if isempty(arg2), error('Usage: show_results(F, L) or show_results(results)'); end
	F = arg1; L = arg2;
	if ~isempty(arg3) && islogical(arg3)
		optsLocal = struct('tuneK', arg3, 'K', 5);
	elseif ~isempty(arg3) && isstruct(arg3)
		optsLocal = arg3;
	else
		optsLocal = struct('tuneK', false, 'K', 5);
	end
	% Use canonical trainer
	results = train_models(F, L, optsLocal);
	labels = L;
end

% Print concise per-model summary
fprintf('\n--- Results Summary ---\n');
for m = 1:numel(results.names)
	acc = results.metrics{m}.accuracy * 100;
	meanF1 = mean(results.metrics{m}.f1);
	fprintf('%s: Accuracy=%.2f%%, MeanF1=%.3f\n', results.names{m}, acc, meanF1);
end

% Decide which confusion matrices to show
modelsToShow = 1;
if opts.showAllConfusions, modelsToShow = 1:numel(results.names); end

outDir = fullfile(pwd,'results_figs');
if opts.saveFigures && ~exist(outDir,'dir'), mkdir(outDir); end

% Load activity labels for better visualization
activityLabels = {};
if exist('labels','var') && ~isempty(labels)
    try
        possiblePaths = {
            fullfile(pwd, 'UCI HAR Dataset', 'activity_labels.txt'),
            fullfile(fileparts(pwd), 'UCI HAR Dataset', 'activity_labels.txt'),
            fullfile(pwd, '..', 'UCI HAR Dataset', 'activity_labels.txt')
        };
        alfile = '';
        for p = 1:numel(possiblePaths)
            if isfile(possiblePaths{p})
                alfile = possiblePaths{p};
                break;
            end
        end
        
        if ~isempty(alfile)
            L = readlines(alfile);
            activityMap = containers.Map('KeyType','double','ValueType','char');
            for kk = 1:numel(L)
                parts = split(strtrim(L(kk)));
                if numel(parts) >= 2
                    k = str2double(parts(1));
                    v = char(strjoin(parts(2:end),' '));
                    activityMap(k) = v;
                end
            end
            classes = unique(labels);
            for c = 1:numel(classes)
                if isKey(activityMap, classes(c))
                    activityLabels{c} = activityMap(classes(c));
                else
                    activityLabels{c} = sprintf('Class %d', classes(c));
                end
            end
        end
    catch
        classes = unique(labels);
        defaultLabels = {'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'};
        for c = 1:numel(classes)
            if classes(c) >= 1 && classes(c) <= 6
                activityLabels{c} = defaultLabels{classes(c)};
            else
                activityLabels{c} = sprintf('Class %d', classes(c));
            end
        end
    end
end

for mi = modelsToShow
	met = results.metrics{mi};
	C = met.confusion;
	
	% Use MATLAB Toolbox functions for professional visualization
	if exist('confusionchart','file') == 2
		% Use confusionchart (Deep Learning Toolbox or Statistics Toolbox)
		fh = figure('Name', results.names{mi}, 'Color', 'white');
		cm = confusionchart(C, activityLabels, 'Colormap', 'sky', 'Title', sprintf('%s - Confusion matrix', results.names{mi}));
		cm.FontSize = 11;
		cm.XLabel = 'Predicted';
		cm.YLabel = 'True';
		fh = gcf;
	else
		% Fallback: Use imagesc with sky colormap
		fh = figure('Name', results.names{mi}, 'Color', 'white');
		imagesc(C);
		colormap('sky');
		colorbar;
		
		nC = size(C,1);
		if ~isempty(activityLabels) && numel(activityLabels) >= nC
			xticklabels(activityLabels(1:nC));
			yticklabels(activityLabels(1:nC));
		else
			classes = unique(labels);
			xticklabels(arrayfun(@num2str, classes, 'UniformOutput', false));
			yticklabels(arrayfun(@num2str, classes, 'UniformOutput', false));
		end
		xticks(1:nC);
		yticks(1:nC);
		
		xlabel('Predicted', 'FontSize', 12, 'FontWeight', 'bold');
		ylabel('True', 'FontSize', 12, 'FontWeight', 'bold');
		title(sprintf('%s - Confusion matrix', results.names{mi}), 'FontSize', 14, 'FontWeight', 'bold');
		
		% Annotate counts
		maxVal = max(C(:));
		for i = 1:nC
			for j = 1:nC
				val = C(i,j);
				if val > maxVal * 0.5
					textColor = 'white';
				else
					textColor = 'black';
				end
				text(j, i, num2str(val), 'HorizontalAlignment', 'center', ...
					'VerticalAlignment', 'middle', 'Color', textColor, ...
					'FontSize', 10, 'FontWeight', 'bold');
			end
		end
		axis square;
	end
	
	if opts.saveFigures
		try
			pngFile = fullfile(outDir, sprintf('%s_confusion.png', results.names{mi}));
			try exportgraphics(fh, pngFile); catch, print(fh, pngFile, '-dpng'); end
			fprintf('Saved confusion PNG to %s\n', pngFile);
		catch
			fprintf('Warning: could not save figure for %s\n', results.names{mi});
		end
	end
	if ~opts.showAllConfusions, break; end
end

% If labels provided, optionally print per-class P/R/F1 for the first model
if exist('labels','var') && ~isempty(labels)
	prec = results.metrics{1}.precision; rec = results.metrics{1}.recall; f1 = results.metrics{1}.f1;
	classes = unique(labels);
	% Decide whether to show per-class metrics. If user explicitly set opts.showPerClass,
	% obey it. Otherwise auto-decide using coefficient-of-variation (std/mean) of per-class F1.
	showPerClass = false;
	if ~isempty(opts.showPerClass)
		showPerClass = logical(opts.showPerClass);
	else
		% compute CV robustly (avoid division by zero)
		meanF1 = mean(f1);
		if meanF1 > 0
			cv = std(f1) / meanF1;
		else
			cv = Inf;
		end
		% If relative variation is small (CV < 0.05), consider per-class metrics stable
		showPerClass = (cv < 0.05);
	end

	if showPerClass
		fprintf('\nPer-class metrics (model: %s):\n', results.names{1});
		for k = 1:numel(classes)
			fprintf('Class %d: P=%.3f, R=%.3f, F1=%.3f\n', classes(k), prec(k), rec(k), f1(k));
		end
	else
		fprintf('\nPer-class metrics suppressed (high variation). To force display, call:\n');
		fprintf('  show_results(..., labels, [], struct(''showPerClass'', true))\n');
	end
end

if opts.summaryOnly
	if nargout >= 1, varargout{1} = results; end
	return;
end

if nargout >= 1, varargout{1} = results; end
end
