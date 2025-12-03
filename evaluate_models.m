function evaluate_models(results, features, labels, opts)
%EVALUATE_MODELS Plot confusion charts and display metrics for trained models
%   evaluate_models(results, features, labels, opts)
% opts.summaryOnly (default false) -> only print accuracy and mean F1, do not
%   generate/save confusion/ROC figures.

if nargin < 4, opts = struct(); end
if ~isfield(opts,'summaryOnly'), opts.summaryOnly = false; end
if ~isfield(opts,'outDir'), opts.outDir = fullfile(pwd,'results_figs'); end
if ~isfield(opts,'overwrite'), opts.overwrite = false; end

outDir = opts.outDir;
if exist(outDir,'dir')==7 && ~opts.overwrite
    % if a run-specific folder was provided, use it; otherwise create a run subfolder
    if isempty(dir(fullfile(outDir,'*')))
        % empty folder, use it
    else
        % if outDir already contains files and not overwrite, ensure unique folder
        % if outDir already looks like a run_ timestamp, keep it
        [~,last] = fileparts(outDir);
        if startsWith(last,'run_')
            % already run-specific, keep using it
        else
            ts = datestr(now,'yyyymmdd_HHMMSS');
            outDir = fullfile(outDir, ['run_' ts]);
            if ~exist(outDir,'dir'), mkdir(outDir); end
        end
    end
else
    if ~exist(outDir,'dir'), mkdir(outDir); end
end

if nargin < 3
    error('Usage: evaluate_models(results, features, labels)');
end

outDir = fullfile(pwd,'results_figs');
if ~exist(outDir,'dir'), mkdir(outDir); end

for m = 1:length(results.models)
    name = results.names{m};
    metrics = results.metrics{m};

    fprintf('\nModel: %s\n', name);
    fprintf('  Accuracy: %.2f%%\n', metrics.accuracy*100);
    fprintf('  Mean F1:  %.3f\n', mean(metrics.f1));

    if opts.summaryOnly
        % summary-only mode: don't generate/save figures
        continue;
    end

    % Confusion matrix using MATLAB Toolbox functions (Statistics and Machine Learning Toolbox)
    % Prefer confusionchart if available, otherwise use confusionmat + imagesc
    C = metrics.confusion;
    if isempty(C) || (~isnumeric(C))
        % Compute confusion matrix from predictions
        try
            preds = metrics.preds;
            % Use MATLAB's confusionmat function (Statistics Toolbox)
            if exist('confusionmat','file') == 2
                C = confusionmat(labels, preds);
            else
                % Fallback: manual computation
                classes = unique(labels);
                C = zeros(numel(classes));
                for i = 1:numel(labels)
                    t = find(classes==labels(i)); p = find(classes==preds(i));
                    if ~isempty(t) && ~isempty(p)
                        C(t,p) = C(t,p) + 1;
                    end
                end
            end
        catch
            fprintf('  Could not compute confusion matrix for model %s; continuing.\n', name);
            C = zeros(length(unique(labels)));
        end
    end
    
    % Load activity labels for better visualization
    activityLabels = {};
    try
        % Try to find activity_labels.txt in common locations
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
                    v = char(strjoin(parts(2:end),' ')); % Keep original format (WALKING, WALKING_UPSTAIRS, etc.)
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
        % Fallback to default labels
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
    
    % If activityLabels is still empty, use class numbers
    if isempty(activityLabels)
        classes = unique(labels);
        for c = 1:numel(classes)
            activityLabels{c} = sprintf('Class %d', classes(c));
        end
    end
    
    % Create confusion matrix visualization using MATLAB Toolbox
    fh = figure('Visible','off','Color','white','InvertHardcopy','off');
    
    % Create consistent blue-style confusion matrix visualization (imagesc)
    try
        % Blue colormap (light -> deep blue)
        blues = interp1([0 1],[1 1 1; 0.06 0.25 0.54],linspace(0,1,256));
        fh = figure('Visible','off','Color','w','Position',[80 80 1400 920]);
        imagesc(C); colormap(blues); hb = colorbar('EastOutside'); hb.FontSize = 12; hb.TickDirection = 'out';
        ax = gca;
        nC = size(C,1);
        ax.XTick = 1:nC; ax.YTick = 1:nC;
        % Use activity labels when available
        if numel(activityLabels) >= nC
            ax.XTickLabel = activityLabels(1:nC);
            ax.YTickLabel = activityLabels(1:nC);
        else
            classes = unique(labels);
            ax.XTickLabel = arrayfun(@num2str, classes, 'UniformOutput', false);
            ax.YTickLabel = arrayfun(@num2str, classes, 'UniformOutput', false);
        end
        ax.XTickLabelRotation = 30; ax.FontSize = 12; ax.TickLength = [0 0];
        title(sprintf('%s - Confusion matrix', name), 'FontSize', 20, 'FontWeight', 'bold', 'Color', [0 0.3 0.6]);
        xlabel('Predicted', 'FontSize', 14, 'Color', [0 0.3 0.6]); ylabel('True', 'FontSize', 14, 'Color', [0 0.3 0.6]);

        % Draw white grid lines between cells
        hold on;
        for r=0:nC
            plot([0.5 nC+0.5],[r+0.5 r+0.5],'Color',[1 1 1],'LineWidth',1.2);
        end
        for c=0:nC
            plot([c+0.5 c+0.5],[0.5 nC+0.5],'Color',[1 1 1],'LineWidth',1.2);
        end

        % Overlay numeric labels in each cell
        mx = max(C(:));
        for i = 1:nC
            for j = 1:nC
                val = C(i,j);
                if mx>50 && abs(round(val)-val)<eps
                    s = sprintf('%d',round(val));
                else
                    s = sprintf('%.2f',val);
                end
                txtColor = [0 0.2 0.45];
                if val > mx*0.6, txtColor = [1 1 1]; end
                text(j, i, s, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', txtColor, 'FontSize', 12, 'FontWeight', 'bold');
            end
        end
        axis tight; axis ij;
    catch ME
        % Fallback to previous behavior if something goes wrong
        warning('Confusion plot failed with error: %s. Reverting to simple imagesc.', ME.message);
        fh = figure('Visible','off'); imagesc(C); colorbar; title(sprintf('%s - Confusion matrix', name));
    end
    % Save confusion figure; if file exists, append timestamp to filename
    fnameBase = fullfile(outDir, sprintf('%s_confusion', name));
    figFile = [fnameBase '.fig']; pngFile = [fnameBase '.png'];
    if exist(figFile,'file')==2 || exist(pngFile,'file')==2
        ts = datestr(now,'yyyymmdd_HHMMSS');
        figFile = fullfile(outDir, sprintf('%s_confusion_%s.fig', name, ts));
        pngFile = fullfile(outDir, sprintf('%s_confusion_%s.png', name, ts));
    end
    savefig(fh, figFile);
    try
        exportgraphics(fh, pngFile);
    catch
        print(fh, pngFile, '-dpng');
    end
    close(fh);

    % Try ROC one-vs-all if scores available
    % Attempt ROC only when necessary functions exist and model supports scoring
    M = results.models{m};
    canPerfcurve = (exist('perfcurve','file')==2);
    canKfoldPredict = true;
    try
        % check if kfoldPredict is available for this model (works for crossval objects)
        canKfoldPredict = exist('kfoldPredict','file')==2;
    catch
        canKfoldPredict = false;
    end
    if canPerfcurve && canKfoldPredict && ~isstruct(M)
        try
            CvM = crossval(M, 'KFold', 5);
            [~,scores] = kfoldPredict(CvM);
            classes = unique(labels);
            fh2 = figure('Visible','off'); hold on;
            for c = 1:length(classes)
                pos = (labels == classes(c));
                [Xroc,Yroc,~,AUC] = perfcurve(pos, scores(:,c), 1);
                plot(Xroc, Yroc, 'DisplayName', sprintf('Class %d (AUC=%.2f)', classes(c), AUC));
            end
            xlabel('False positive rate'); ylabel('True positive rate');
            title(sprintf('%s ROC (one-vs-all)', name)); legend('Location','best');
            % Save ROC figure, avoid overwriting
            rocBase = fullfile(outDir, sprintf('%s_ROC', name));
            rocFig = [rocBase '.fig']; rocPng = [rocBase '.png'];
            if exist(rocFig,'file')==2 || exist(rocPng,'file')==2
                ts = datestr(now,'yyyymmdd_HHMMSS');
                rocFig = fullfile(outDir, sprintf('%s_ROC_%s.fig', name, ts));
                rocPng = fullfile(outDir, sprintf('%s_ROC_%s.png', name, ts));
            end
            savefig(fh2, rocFig);
            try
                exportgraphics(fh2, rocPng);
            catch
            end
            close(fh2);
        catch ME
            fprintf('  ROC: could not be computed for %s (%s).\n', name, ME.message);
        end
    else
        fprintf('  ROC: skipped (perfcurve/crossval/kfoldPredict not available or model not compatible).\n');
    end
end

fprintf('\nAll figures saved into %s.\n', outDir);
end
