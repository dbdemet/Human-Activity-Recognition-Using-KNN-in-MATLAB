function app = create_app_clean()
% CREATE_APP_CLEAN Simplified demo UI for UCI HAR dataset (single-file).

% Create the UIFigure hidden to avoid multiple redraws which can slow startup
app.UIFigure = uifigure('Name','UCI HAR Demo (clean)','Position',[200 200 980 560],'Visible','off');

% Performance defaults (tunable)
NfeatDefault = 8;      % use very small feature subset for plotting (aggressive)
NpredictDefault = 30;   % small synchronous cache size - stratified (5 per class for 6 classes)
NquickDefault = 120;    % background quick cache size - stratified (20 per class for 6 classes)
targetPlotPointsDefault = 40; % points to draw for each signal (downsampled)

% Controls panel
app.ControlPanel = uipanel(app.UIFigure,'Title','','Position',[10 462 960 86],'BackgroundColor',[0.98 0.98 0.98]);
btnY = 14; btnH = 56; btnW = 120; gap = 12; x = 12;
app.LoadButton = uibutton(app.ControlPanel,'push','Position',[x btnY btnW btnH],'Text','Load Data','ButtonPushedFcn',@(~,~)onLoad()); x = x + btnW + gap;
app.ExtractButton = uibutton(app.ControlPanel,'push','Position',[x btnY btnW btnH],'Text','Extract Features','ButtonPushedFcn',@(~,~)onExtract()); x = x + btnW + gap;
app.DemoButton = uibutton(app.ControlPanel,'push','Position',[x btnY btnW btnH],'Text','Quick Demo','ButtonPushedFcn',@(~,~)onDemo()); x = x + btnW + gap;
% style buttons
for b = [app.LoadButton app.ExtractButton app.DemoButton]
    b.FontWeight = 'bold'; b.FontSize = 12; b.BackgroundColor = [0.92 0.92 1.0]; b.Tooltip = 'Click to perform action';
end

app.SampleLabel = uilabel(app.ControlPanel,'Position',[x 36 50 22],'Text','Sample');
% create a rounded-looking background frame (overlay) and place the edit on top
% numeric edit field (no spinner arrows); make it visually compact and close to label
app.SampleEdit = uieditfield(app.ControlPanel,'numeric','Position',[x+60 34 64 26],'Value',1,'Limits',[1 Inf],'ValueChangedFcn',@(src,~)onSampleChange(src));
app.SampleEdit.FontSize = 11; app.SampleEdit.FontWeight = 'bold';
% soften visible border by matching background and padding; keep single box (no duplicate frame)
try
    app.SampleEdit.BackgroundColor = [1 1 1];
    % small padding: increase width slightly so it visually aligns with others
    app.SampleEdit.Position(3) = 64;
    % round corners for sample edit field
    app.SampleEdit.EdgeColor = [0.7 0.7 0.7];
catch
end
% add spacing between Sample and Set controls (gap of 20 pixels)
xSet = x + 60 + 64 + 20; % SampleEdit right edge + gap
app.SetLabel = uilabel(app.ControlPanel,'Position',[xSet 36 30 22],'Text','Set');
app.SetDrop = uidropdown(app.ControlPanel,'Position',[xSet+30 34 92 26],'Items',{'test','train'},'ValueChangedFcn',@(~,~)onExtract());

% Place prediction button right after Set dropdown (replacing Model dropdown)
setRightX = xSet + 30 + 92; % SetDrop X + width
gapPred = 20;
predX = setRightX + gapPred;
predW = min(420, max(200, 960 - predX - 12)); predH = 30; predY = 32;
% Create prediction button to display results
app.PredButton = uibutton(app.ControlPanel,'push','Position',[predX predY predW predH],'Text','Predicted: -', 'FontWeight','bold','FontSize',13,'HorizontalAlignment','left');
app.PredButton.BackgroundColor = [1 1 1]; app.PredButton.FontColor = [0 0 0];
% make button non-obtrusive (no callback) and bring to front
try
    app.PredButton.ButtonPushedFcn = [];
    uistack(app.PredButton,'top');
catch
end

% Axes
app.AxLeft = uiaxes(app.UIFigure,'Position',[12 62 440 380]);
app.AxRight = uiaxes(app.UIFigure,'Position',[468 62 440 380]);
app.AxLeft.Title.String = 'Raw signals'; app.AxLeft.Title.FontSize = 14;
app.AxRight.Title.String = 'Normalized features (z-score)'; app.AxRight.Title.FontSize = 14;

% storage
app.UserData = struct('dataDir','', 'cache',[],'featCache',[],'activityMap',[],'lastPred',[],'doPrecompute',false,'Nfeat',NfeatDefault);

%% Callbacks
    function onLoad()
        d = uigetdir(pwd,'Select UCI HAR Dataset folder');
        if isequal(d,0), return; end
        app.UserData.dataDir = d;
        app.PredButton.Text = 'Predicted: -'; app.PredButton.BackgroundColor = [1 1 1]; app.PredButton.FontColor = [0 0 0];
        app.UserData.cache = []; app.UserData.featCache = [];
        app.UserData.activityMap = []; % Reset activity map to reload labels

        % Preload train cache in background for faster predictions (stratified)
        try
            yfile = fullfile(app.UserData.dataDir,'train','y_train.txt');
            if isfile(yfile)
                ytrain = readmatrix(yfile);
                nUse = numel(ytrain);
                % Build a small initial stratified cache immediately for fast first prediction
                Ninit = min(NpredictDefault, nUse);
                try
                    app.UserData.featCache = struct();
                    [F, indices] = build_feat_cache_stratified('train', Ninit);
                    app.UserData.featCache.train = F;
                    app.UserData.featCache.trainIndices = indices;
                catch
                end
                % Build larger cache in background
                if ~isfield(app.UserData,'trainCacheTimer') || isempty(app.UserData.trainCacheTimer)
                    t = timer('ExecutionMode','singleShot','StartDelay',0.3,'TimerFcn',@(~,~)setTrainCache());
                    app.UserData.trainCacheTimer = t;
                    start(t);
                end
                % Print sample mapping for user reference
                print_sample_activity_mapping();
            end
        catch
        end
    end

    function setTrainCache()
        % runs in timer context to compute larger train features cache without blocking UI (stratified)
        try
            if isempty(app.UserData.dataDir), return; end
            yfile = fullfile(app.UserData.dataDir,'train','y_train.txt');
            if ~isfile(yfile), return; end
            ytrain = readmatrix(yfile); nUse = numel(ytrain);
            % Build larger stratified cache in background (improves prediction accuracy)
            Nfast = min(nUse, NquickDefault);
            [F, indices] = build_feat_cache_stratified('train', Nfast);
            % set into app.UserData safely
            if ~isfield(app.UserData,'featCache') || isempty(app.UserData.featCache), app.UserData.featCache = struct(); end
            app.UserData.featCache.train = F;
            app.UserData.featCache.trainIndices = indices;
        catch
        end
    end

    function onSampleChange(src)
        val = max(1, round(src.Value)); src.Value = val; onExtract();
    end

    function onExtract()
        if isempty(app.UserData.dataDir)
            uialert(app.UIFigure,'Please load the UCI HAR Dataset folder first.','Error');
            return;
        end
        sampleIdx = max(1, round(app.SampleEdit.Value)); setType = app.SetDrop.Value;
        % reset prediction display
        app.PredButton.Text = 'Predicted: -'; app.PredButton.BackgroundColor = [1 1 1]; app.PredButton.FontColor = [0 0 0];
        drawnow;

        % STEP 1: Compute features for current sample (fast, single sample)
        if isfield(app.UserData,'featCache') && isstruct(app.UserData.featCache) && isfield(app.UserData.featCache,setType) && size(app.UserData.featCache.(setType),1) >= sampleIdx
            Xs = app.UserData.featCache.(setType)(sampleIdx,:);
        else
            try
                % compute only this sample synchronously (fast)
                Xs = compute_features_for_sample(app.UserData.dataDir, setType, sampleIdx);
            catch ME
                uialert(app.UIFigure,['Feature extraction failed: ' ME.message],'Error'); return;
            end
        end

        % limit features used for plotting & distance computations for speed
        Nfeat = min(NfeatDefault, numel(Xs));
        Xs_tr = Xs(1:Nfeat);
        
        % STEP 2 & 3: Plot raw signals and normalized features simultaneously (fast, immediate visual feedback)
        ax1 = app.AxLeft; ax2 = app.AxRight;
        signals = {'body_acc_x','body_acc_y','body_acc_z'};
        
        % Prepare raw signals data
        cla(ax1); hold(ax1,'on'); maxAbs = 0;
        signalData = cell(3,1);
        for s = 1:3
            fname = fullfile(app.UserData.dataDir,setType,'Inertial Signals',[signals{s} '_' setType '.txt']);
            if isfield(app.UserData,'cache') && isstruct(app.UserData.cache) && isfield(app.UserData.cache,setType) && isfield(app.UserData.cache.(setType),signals{s})
                M = app.UserData.cache.(setType).(signals{s});
            else
                M = readmatrix(fname);
                if ~isfield(app.UserData,'cache') || isempty(app.UserData.cache), app.UserData.cache = struct(); end
                if ~isfield(app.UserData.cache,setType) || isempty(app.UserData.cache.(setType)), app.UserData.cache.(setType) = struct(); end
                app.UserData.cache.(setType).(signals{s}) = M;
            end
            rowFull = double(M(sampleIdx,:)); rowFull = rowFull - mean(rowFull);
            % downsample for plotting to speed UI (keep representative waveform)
            targetPoints = targetPlotPointsDefault; step = max(1, ceil(length(rowFull)/targetPoints));
            idxs = 1:step:length(rowFull);
            signalData{s} = struct('x',idxs,'y',rowFull(idxs));
            maxAbs = max(maxAbs, max(abs(rowFull)));
        end
        
        % Prepare normalized features data — prefer full feature vector for visualization
        Xfull = Xs; % full feature vector for plotting
        if isfield(app.UserData,'featCache') && isstruct(app.UserData.featCache) && isfield(app.UserData.featCache,setType) && ~isempty(app.UserData.featCache.(setType))
            Fmat = app.UserData.featCache.(setType);
            % if training cache has fewer cols than Xfull, fall back to local normalization
            if size(Fmat,2) == numel(Xfull)
                mu = mean(Fmat,1); sigma = std(Fmat,[],1)+eps; Xnorm = (Xfull - mu)./sigma;
            else
                % compute small neighborhood features to normalize full vector
                Nnorm = 8; smallF = [];
                L = max(1, sampleIdx - floor(Nnorm/2)); R = L + Nnorm - 1;
                for ii = L:R
                    try
                        smallF(end+1,:) = compute_features_for_sample(app.UserData.dataDir, setType, ii); %#ok<AGROW>
                    catch
                    end
                end
                if ~isempty(smallF) && size(smallF,2) == numel(Xfull)
                    mu = mean(smallF,1); sigma = std(smallF,[],1)+eps; Xnorm = (Xfull - mu)./sigma;
                else
                    Xnorm = (Xfull - mean(Xfull))./(std(Xfull)+eps);
                end
            end
        else
            % small local normalization sample for quick plotting (full features)
            Nnorm = 8; smallF = [];
            L = max(1, sampleIdx - floor(Nnorm/2)); R = L + Nnorm - 1;
            for ii = L:R
                try
                    smallF(end+1,:) = compute_features_for_sample(app.UserData.dataDir, setType, ii); %#ok<AGROW>
                catch
                end
            end
            if ~isempty(smallF)
                mu = mean(smallF,1); sigma = std(smallF,[],1)+eps; Xnorm = (Xfull - mu)./sigma;
            else
                Xnorm = (Xfull - mean(Xfull))./(std(Xfull)+eps);
            end
        end
        
        % Plot both simultaneously
        for s = 1:3
            plot(ax1, signalData{s}.x, signalData{s}.y, 'LineWidth', 1);
        end
        hold(ax1,'off'); if maxAbs>0, ylim(ax1,[-maxAbs maxAbs]*1.1); end
        legend(ax1,{'acc_x','acc_y','acc_z'}); title(ax1,sprintf('Raw signals (sample %d)',sampleIdx));
        
        cla(ax2); bar(ax2, Xnorm); title(ax2,'Normalized features (z-score)'); xlabel(ax2,'Feature index');
        drawnow; % Update UI immediately after plotting both graphs
        
        % STEP 4: Prediction (immediately after plots, using existing cache or building minimal cache)
        try
            yfile = fullfile(app.UserData.dataDir,'train','y_train.txt');
            if isfile(yfile)
                ytrainFull = readmatrix(yfile); nUse = numel(ytrainFull);
                hasTrainCache = isfield(app.UserData,'featCache') && isstruct(app.UserData.featCache) && isfield(app.UserData.featCache,'train') && ~isempty(app.UserData.featCache.train) && size(app.UserData.featCache.train,1) >= min(10, nUse);
                
                if hasTrainCache
                    % Use existing cache for instant prediction
                    XtrainFull = app.UserData.featCache.train;
                    % Use all features for better accuracy
                    Xtrain = XtrainFull;
                    % Get corresponding labels from cache indices
                    if isfield(app.UserData.featCache,'trainIndices') && ~isempty(app.UserData.featCache.trainIndices)
                        ytrain = ytrainFull(app.UserData.featCache.trainIndices);
                    else
                        % Fallback: use first N samples
                        nUse2 = size(Xtrain,1);
                        ytrain = ytrainFull(1:nUse2);
                    end
                else
                    % Build minimal stratified cache synchronously for immediate prediction
                    Npredict = min(NpredictDefault, nUse);
                    [tmpSmall, trainIndices] = build_feat_cache_stratified('train', Npredict);
                    if ~isfield(app.UserData,'featCache') || isempty(app.UserData.featCache), app.UserData.featCache = struct(); end
                    app.UserData.featCache.train = tmpSmall;
                    app.UserData.featCache.trainIndices = trainIndices;
                    Xtrain = app.UserData.featCache.train;
                    ytrain = ytrainFull(trainIndices);
                    
                    % Start background timer to build larger cache (non-blocking, improves future predictions)
                    try
                        if ~isfield(app.UserData,'trainQuickTimer') || isempty(app.UserData.trainQuickTimer)
                            Nquick = min(NquickDefault, nUse);
                            tquick = timer('ExecutionMode','singleShot','StartDelay',0.05,'TimerFcn',@(~,~)backgroundBuildQuick(Nquick,setType));
                            app.UserData.trainQuickTimer = tquick; start(tquick);
                        elseif strcmp(get(app.UserData.trainQuickTimer,'Running'),'off')
                            Nquick = min(NquickDefault, nUse);
                            try stop(app.UserData.trainQuickTimer); delete(app.UserData.trainQuickTimer); catch, end
                            tquick = timer('ExecutionMode','singleShot','StartDelay',0.05,'TimerFcn',@(~,~)backgroundBuildQuick(Nquick,setType));
                            app.UserData.trainQuickTimer = tquick; start(tquick);
                        end
                    catch
                    end
                end
                
                % Fast nearest-centroid prediction using ALL features for better accuracy
                try
                    % Use full feature vector for prediction
                    Xs_pred = Xs; % use all features, not truncated
                    if size(Xtrain,2) ~= numel(Xs_pred)
                        % Align dimensions
                        minDim = min(size(Xtrain,2), numel(Xs_pred));
                        Xtrain = Xtrain(:,1:minDim);
                        Xs_pred = Xs_pred(1:minDim);
                    end
                    
                    classes = unique(ytrain);
                    if numel(classes) < 2
                        ypred = classes(1);
                    else
                        centroids = zeros(numel(classes), size(Xtrain,2));
                        for ci = 1:numel(classes)
                            idxc = ytrain==classes(ci);
                            if any(idxc)
                                centroids(ci,:) = mean(Xtrain(idxc,:),1);
                            end
                        end
                        difc = centroids - repmat(Xs_pred, size(centroids,1),1);
                        Dc = sqrt(sum(difc.^2,2)); 
                        [~, ciidx] = min(Dc); 
                        ypred = classes(ciidx);
                    end
                catch
                    ypred = mode(ytrain);
                end
                
                aname = upper(get_activity_name(ypred));
                app.PredButton.Text = ['Predicted: ' aname]; 
                app.PredButton.BackgroundColor = [0.88 1.0 0.88]; 
                app.PredButton.FontColor = [0 0.4 0];
                try uistack(app.PredButton,'top'); catch, end
                app.UserData.lastPred = struct('class',double(ypred),'name',aname,'sample',sampleIdx,'set',setType);
                drawnow; % Ensure prediction appears immediately (within 1-2 seconds of Extract click)
            end
        catch ME
            % Silent fail - prediction is optional for UI responsiveness
        end
    end

    function finishTrainCacheAndPredict(Xs, setType, sampleIdx)
        try
            if isempty(app.UserData.dataDir), return; end
            yfile = fullfile(app.UserData.dataDir,'train','y_train.txt'); if ~isfile(yfile), return; end
            ytrain = readmatrix(yfile); nUse = numel(ytrain);
            % build full train cache (may take time)
            app.UserData.featCache.train = build_feat_cache('train', nUse);
            % now compute prediction using nearest centroid (fast and consistent)
            Xtrain = app.UserData.featCache.train; if isempty(Xtrain), return; end
            nUse2 = min(size(Xtrain,1), nUse); Xtrain = Xtrain(1:nUse2,:); ytrain = ytrain(1:nUse2);
            classes = unique(ytrain); centroids = zeros(numel(classes), size(Xtrain,2));
            for ci = 1:numel(classes), centroids(ci,:) = mean(Xtrain(ytrain==classes(ci),:),1); end
            difc = centroids - repmat(Xs, size(centroids,1),1); Dc = sqrt(sum(difc.^2,2)); [~, ci] = min(Dc); ypred = classes(ci);
            aname = upper(get_activity_name(ypred));
            % update UI
            app.PredButton.Text = ['Predicted: ' aname];
            app.PredButton.BackgroundColor = [0.88 1.0 0.88]; app.PredButton.FontColor = [0 0.4 0];
            try uistack(app.PredButton,'top'); catch, end
            app.UserData.lastPred = struct('class',double(ypred),'name',aname,'sample',sampleIdx,'set',setType);
        catch
        end
    end

    function backgroundBuildQuick(Nquick,setType)
        % builds a slightly larger quick train cache in background timer (stratified)
        try
            if isempty(app.UserData.dataDir), return; end
            [tmp, indices] = build_feat_cache_stratified('train', Nquick);
            if ~isfield(app.UserData,'featCache') || isempty(app.UserData.featCache), app.UserData.featCache = struct(); end
            app.UserData.featCache.train = tmp;
            app.UserData.featCache.trainIndices = indices;
        catch
        end
    end

    function onDemo()
        % Quick Demo: call the same Extract routine so predictions match
        if isempty(app.UserData.dataDir)
            uialert(app.UIFigure,'Please load the UCI HAR Dataset folder first.','Error'); return;
        end
        onExtract();
        % ensure styling applied (in case onExtract reset it)
        if isfield(app.UserData,'lastPred') && ~isempty(app.UserData.lastPred)
            lp = app.UserData.lastPred;
            app.PredButton.Text = ['Predicted: ' lp.name];
            app.PredButton.BackgroundColor = [0.88 1.0 0.88];
            app.PredButton.FontColor = [0 0.4 0];
            try uistack(app.PredButton, 'top'); catch, end
        end
        % highlight the selected sample's index column in red (sample index = feature index)
        try
            axRight = app.AxRight; bars = findall(axRight,'Type','Bar');
            if ~isempty(bars)
                b = bars(1); 
                sampleIdx = max(1, round(app.SampleEdit.Value));
                % Feature index corresponds to sample index (1-based)
                % But we only show Nfeat features, so map sample index to visible feature range
                numBars = numel(b.YData);
                if numBars > 0
                    % Use sample index modulo number of bars to highlight corresponding feature
                    featIdx = mod(sampleIdx - 1, numBars) + 1;
                    featIdx = min(max(1, featIdx), numBars); % ensure within bounds [1, numBars]
                    cmap = repmat([0.2 0.6 0.8], numBars, 1); 
                    cmap(featIdx, :) = [0.9 0.2 0.2]; % red for selected sample's feature index
                    try
                        b.FaceColor = 'flat'; 
                        b.CData = cmap;
                    catch
                    end
                end
            end
        catch
        end
    end

%% Helpers
    function feat = compute_features_for_sample(dataDir, setType, sampleIdx)
        fs = 50;
        signals = {'body_acc_x','body_acc_y','body_acc_z', 'body_gyro_x','body_gyro_y','body_gyro_z', 'total_acc_x','total_acc_y','total_acc_z'};
        featPerSig = 8; feat = zeros(1, numel(signals)*featPerSig);
        for s = 1:numel(signals)
            fname = fullfile(dataDir,setType,'Inertial Signals',[signals{s} '_' setType '.txt']);
            if ~isfile(fname), error('Expected file not found: %s', fname); end
            M = readmatrix(fname);
            if sampleIdx > size(M,1), error('Sample index %d out of range for file %s', sampleIdx, fname); end
            x = double(M(sampleIdx,:));
            mu = mean(x); sd = std(x); med = median(x); rmsv = sqrt(mean(x.^2)); energy = sum(x.^2);
            zcr = sum(abs(diff(x>0)))/length(x);
            NFFT = length(x); freqs = (0:NFFT-1)*(fs/NFFT);
            X = abs(fft(x,NFFT)).^2; P = X(1:floor(NFFT/2)+1); f = freqs(1:floor(NFFT/2)+1);
            bp_slow = sum(P(f >= 0.1 & f <= 3)); bp_mid = sum(P(f > 3 & f <= 12));
            base = (s-1)*featPerSig;
            feat(base+1:base+8) = [mu sd med rmsv energy zcr bp_slow bp_mid];
        end
    end

    function [F, indices] = build_feat_cache_stratified(setType, Nmax)
        % Build feature cache with stratified sampling (returns features and indices)
        if isempty(app.UserData.dataDir), error('Data folder not set'); end
        signals = {'body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z'};
        mats = cell(1,numel(signals)); minRows = inf;
        for si = 1:numel(signals)
            fname = fullfile(app.UserData.dataDir,setType,'Inertial Signals',[signals{si} '_' setType '.txt']);
            if ~isfile(fname), error('Missing file: %s', fname); end
            M = readmatrix(fname); mats{si} = double(M); minRows = min(minRows, size(M,1));
        end
        
        % Stratified sampling: ensure all classes are represented
        idxs = [];
        if strcmp(setType, 'train') && Nmax < minRows
            try
                yfile = fullfile(app.UserData.dataDir, setType, 'y_train.txt');
                if isfile(yfile)
                    y = readmatrix(yfile);
                    classes = unique(y);
                    nClasses = numel(classes);
                    nPerClass = max(1, floor(Nmax / nClasses)); % samples per class
                    idxs = [];
                    for c = 1:nClasses
                        classIdx = find(y == classes(c));
                        if numel(classIdx) > 0
                            if numel(classIdx) <= nPerClass
                                idxs = [idxs; classIdx(:)];
                            else
                                sel = classIdx(randperm(numel(classIdx), nPerClass));
                                idxs = [idxs; sel(:)];
                            end
                        end
                    end
                    idxs = idxs(1:min(Nmax, numel(idxs)));
                    idxs = sort(idxs);
                end
            catch
                idxs = round(linspace(1, minRows, min(Nmax, minRows)));
            end
        end
        
        if isempty(idxs)
            N = min(Nmax, minRows);
            if N >= 1
                idxs = round(linspace(1, minRows, N));
            else
                idxs = [];
            end
        end
        
        indices = idxs;
        N = numel(idxs);
        if N == 0, F = []; return; end
        
        % preallocate feature matrix
        tmpFeat = compute_features_for_sample(app.UserData.dataDir, setType, idxs(1));
        F = zeros(N, numel(tmpFeat));
        fs = 50;
        featPerSig = 8;
        
        for si = 1:numel(signals)
            M = mats{si}(idxs,:);
            mu = mean(M,2);
            sd = std(M,0,2);
            med = median(M,2);
            rmsv = sqrt(mean(M.^2,2));
            energy = sum(M.^2,2);
            zrc = sum(abs(diff(M>0,1,2)),2) ./ size(M,2);
            NFFT = size(M,2);
            freqs = (0:NFFT-1)*(fs/NFFT);
            X = abs(fft(M,NFFT,2)).^2;
            P = X(:,1:floor(NFFT/2)+1);
            f = freqs(1:floor(NFFT/2)+1);
            idxSlow = (f >= 0.1 & f <= 3);
            idxMid = (f > 3 & f <= 12);
            if any(idxSlow)
                bp_slow = sum(P(:, idxSlow), 2);
            else
                bp_slow = zeros(N,1);
            end
            if any(idxMid)
                bp_mid = sum(P(:, idxMid), 2);
            else
                bp_mid = zeros(N,1);
            end
            base = (si-1)*featPerSig;
            F(:, base + 1) = mu;
            F(:, base + 2) = sd;
            F(:, base + 3) = med;
            F(:, base + 4) = rmsv;
            F(:, base + 5) = energy;
            F(:, base + 6) = zrc;
            F(:, base + 7) = bp_slow;
            F(:, base + 8) = bp_mid;
        end
    end

    function F = build_feat_cache(setType, Nmax)
        if isempty(app.UserData.dataDir), error('Data folder not set'); end
        signals = {'body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z'};
        mats = cell(1,numel(signals)); minRows = inf;
        for si = 1:numel(signals)
            fname = fullfile(app.UserData.dataDir,setType,'Inertial Signals',[signals{si} '_' setType '.txt']);
            if ~isfile(fname), error('Missing file: %s', fname); end
            M = readmatrix(fname); mats{si} = double(M); minRows = min(minRows, size(M,1));
        end
        
        % Stratified sampling: ensure all classes are represented
        idxs = [];
        if strcmp(setType, 'train') && Nmax < minRows
            % For train set, use stratified sampling to get samples from all classes
            try
                yfile = fullfile(app.UserData.dataDir, setType, 'y_train.txt');
                if isfile(yfile)
                    y = readmatrix(yfile);
                    classes = unique(y);
                    nClasses = numel(classes);
                    nPerClass = max(1, floor(Nmax / nClasses)); % samples per class
                    idxs = [];
                    for c = 1:nClasses
                        classIdx = find(y == classes(c));
                        if numel(classIdx) > 0
                            % Randomly select nPerClass samples from this class
                            if numel(classIdx) <= nPerClass
                                idxs = [idxs; classIdx(:)];
                            else
                                sel = classIdx(randperm(numel(classIdx), nPerClass));
                                idxs = [idxs; sel(:)];
                            end
                        end
                    end
                    idxs = idxs(1:min(Nmax, numel(idxs))); % limit to Nmax
                    idxs = sort(idxs); % keep sorted for cache consistency
                end
            catch
                % Fallback to linear sampling if stratified fails
                idxs = round(linspace(1, minRows, min(Nmax, minRows)));
            end
        end
        
        % If stratified sampling didn't work or not train set, use linear sampling
        if isempty(idxs)
            N = min(Nmax, minRows);
            if N >= 1
                idxs = round(linspace(1, minRows, N));
            else
                idxs = [];
            end
        end
        
        N = numel(idxs);
        if N == 0, F = []; return; end
        
        % preallocate feature matrix
        tmpFeat = compute_features_for_sample(app.UserData.dataDir, setType, idxs(1));
        F = zeros(N, numel(tmpFeat));
        fs = 50;
        featPerSig = 8;
        
        % process each signal in vectorized form (per-signal computations on N rows)
        for si = 1:numel(signals)
            M = mats{si}(idxs,:); % N x T (rows selected)
            % basic stats per row
            mu = mean(M,2);
            sd = std(M,0,2);
            med = median(M,2);
            rmsv = sqrt(mean(M.^2,2));
            energy = sum(M.^2,2);
            % zero-crossing rate per row
            zrc = sum(abs(diff(M>0,1,2)),2) ./ size(M,2);
            % frequency band power via FFT along each row
            NFFT = size(M,2);
            freqs = (0:NFFT-1)*(fs/NFFT);
            X = abs(fft(M,NFFT,2)).^2; % N x NFFT
            P = X(:,1:floor(NFFT/2)+1);
            f = freqs(1:floor(NFFT/2)+1);
            idxSlow = (f >= 0.1 & f <= 3);
            idxMid = (f > 3 & f <= 12);
            if any(idxSlow)
                bp_slow = sum(P(:, idxSlow), 2);
            else
                bp_slow = zeros(N,1);
            end
            if any(idxMid)
                bp_mid = sum(P(:, idxMid), 2);
            else
                bp_mid = zeros(N,1);
            end
            % assemble into columns
            base = (si-1)*featPerSig;
            F(:, base + 1) = mu;
            F(:, base + 2) = sd;
            F(:, base + 3) = med;
            F(:, base + 4) = rmsv;
            F(:, base + 5) = energy;
            F(:, base + 6) = zrc;
            F(:, base + 7) = bp_slow;
            F(:, base + 8) = bp_mid;
        end
    end

    function name = get_activity_name(idx)
        name = sprintf('class %d', idx);
        try
            if ~isfield(app.UserData,'dataDir') || isempty(app.UserData.dataDir), return; end
            if isempty(app.UserData.activityMap)
                alfile = fullfile(app.UserData.dataDir,'activity_labels.txt');
                m = containers.Map('KeyType','double','ValueType','char');
                if isfile(alfile)
                    L = readlines(alfile);
                    for kk = 1:numel(L)
                        parts = split(strtrim(L(kk)));
                        if numel(parts) >= 2
                            k = str2double(parts(1)); 
                            v = lower(char(strjoin(parts(2:end),' '))); 
                            v = strrep(v,'_',' '); 
                            % Normalize label names to match user's expected format
                            v = strtrim(v);
                            m(k) = v;
                        end
                    end
                else
                    % Default mapping: 1=walking, 2=walking upstairs, 3=walking downstairs, 4=sitting, 5=standing, 6=laying
                    defaultNames = {'walking','walking upstairs','walking downstairs','sitting','standing','laying'};
                    for kk = 1:numel(defaultNames), m(kk) = defaultNames{kk}; end
                end
                app.UserData.activityMap = m;
            end
            if isKey(app.UserData.activityMap,double(idx)), name = char(app.UserData.activityMap(double(idx))); end
        catch
        end
    end
    
    function print_sample_activity_mapping()
        % Helper function to print sample numbers for each activity (for user reference)
        try
            if isempty(app.UserData.dataDir), return; end
            yfile = fullfile(app.UserData.dataDir,'train','y_train.txt');
            if ~isfile(yfile), return; end
            ytrain = readmatrix(yfile);
            classes = unique(ytrain);
            fprintf('\n=== Sample Numbers by Activity (TRAIN set) ===\n');
            for c = 1:numel(classes)
                classIdx = find(ytrain == classes(c));
                aname = get_activity_name(classes(c));
                fprintf('%s (class %d): Samples %d-%d (first 10: %s)\n', ...
                    upper(aname), classes(c), min(classIdx), max(classIdx), ...
                    mat2str(classIdx(1:min(10,numel(classIdx)))'));
            end
            fprintf('==============================================\n\n');
        catch
        end
    end

% Make the UI visible now that construction finished to avoid multiple redraws
try
    app.UIFigure.Visible = 'on';
catch
end

end
