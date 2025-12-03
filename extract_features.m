function [X, Y] = extract_features(dataDir, setType)
% EXTRACT_FEATURES Minimal feature extractor for UCI HAR (toolbox-free)
%   [X,Y] = extract_features(dataDir) returns features and labels for the
%   train split inside the provided dataset folder (expects standard UCI
%   HAR layout). Optionally pass setType = 'train' or 'test'.
if nargin < 2, setType = 'train'; end
if nargin < 1 || isempty(dataDir), dataDir = pwd; end
% normalize path
dataDir = char(dataDir);

% If the provided path doesn't contain expected files, try common alternatives
sigTestFile = fullfile(dataDir, setType, 'Inertial Signals', ['body_acc_x_' setType '.txt']);
if ~isfile(sigTestFile)
    cwd = pwd;
    % Detect absolute paths to avoid concatenating them with cwd
    isAbs = false;
    if ispc
        % Windows absolute: starts with 'C:\' or UNC '\\'
        if ~isempty(regexp(dataDir, '^[A-Za-z]:\\', 'once')) || strncmp(dataDir, '\\', 2)
            isAbs = true;
        end
    else
        % Unix-like absolute path starts with '/'
        if strncmp(dataDir, '/', 1)
            isAbs = true;
        end
    end

    tried = {};
    found = false;
    % Candidate strategy: if absolute, try the path and its parents; if relative, try cwd-relative variants
    if isAbs
        candidates = { dataDir, fileparts(dataDir), fileparts(fileparts(dataDir)) };
    else
        candidates = { dataDir, fullfile(cwd,dataDir), fullfile(cwd,'..',dataDir), fullfile(fileparts(cwd),dataDir), fullfile(fileparts(fileparts(cwd)),dataDir) };
    end

    % Also try common child folders (one level) under each candidate
    extended = candidates;
    for ci = 1:numel(candidates)
        cand0 = char(candidates{ci});
        if isempty(cand0), continue; end
        try
            D = dir(cand0);
            for di = 1:numel(D)
                if D(di).isdir && ~strcmp(D(di).name,'.') && ~strcmp(D(di).name,'..')
                    child = fullfile(cand0, D(di).name);
                    % add if not already present
                    if ~any(cellfun(@(c) strcmp(c, child), extended))
                        extended{end+1} = child; %#ok<AGROW>
                    end
                end
            end
        catch
            % ignore inaccessible directories
        end
        % Also try the explicit 'UCI HAR Dataset' child if present
        uciChild = fullfile(cand0, 'UCI HAR Dataset');
        if isfolder(uciChild) && ~any(cellfun(@(c) strcmp(c, uciChild), extended))
            extended{end+1} = uciChild; %#ok<AGROW>
        end
    end

    for ci = 1:numel(extended)
        cand = char(extended{ci});
        if isempty(cand), continue; end
        tried{end+1} = cand; %#ok<AGROW>
        testF = fullfile(cand, setType, 'Inertial Signals', ['body_acc_x_' setType '.txt']);
        if isfile(testF)
            dataDir = cand;
            found = true;
            break;
        end
    end
    if ~found
        % nothing found — provide clear error with tried locations
        msg = sprintf('Missing expected file: %s\nTried these locations:\n', sigTestFile);
        for i=1:numel(tried), msg = [msg sprintf(' - %s\n', tried{i})]; end
        error(msg);
    end
end
signals = {'body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z'};
featPerSig = 8;
% read label file if exists
yfile = fullfile(dataDir, setType, ['y_' setType '.txt']);
if isfile(yfile)
    Y = readmatrix(yfile);
else
    Y = [];
end
% read inertial signal files
Mats = cell(1,numel(signals)); minRows = inf;
for s = 1:numel(signals)
    fname = fullfile(dataDir, setType, 'Inertial Signals', [signals{s} '_' setType '.txt']);
    if ~isfile(fname)
        error('Missing expected file: %s', fname);
    end
    M = readmatrix(fname);
    Mats{s} = double(M);
    minRows = min(minRows, size(M,1));
end
N = minRows;
if ~isempty(Y), N = min(N, numel(Y)); end
X = zeros(N, numel(signals)*featPerSig);
for n = 1:N
    rowFeat = zeros(1, numel(signals)*featPerSig);
    idx = 1;
    for s = 1:numel(signals)
        x = Mats{s}(n,:);
        mu = mean(x); sd = std(x); med = median(x); rmsv = sqrt(mean(x.^2)); energy = sum(x.^2);
        zcr = sum(abs(diff(x>0)))/length(x);
        NFFT = length(x); freqs = (0:NFFT-1)*(50/NFFT);
        Xf = abs(fft(x,NFFT)).^2; P = Xf(1:floor(NFFT/2)+1); f = freqs(1:floor(NFFT/2)+1);
        bp_slow = sum(P(f >= 0.1 & f <= 3)); bp_mid = sum(P(f > 3 & f <= 12));
        rowFeat(idx:idx+7) = [mu sd med rmsv energy zcr bp_slow bp_mid]; idx = idx + 8;
    end
    X(n,:) = rowFeat;
end
% trim labels if necessary
if ~isempty(Y) && size(Y,1) >= N
    Y = Y(1:N);
end
end
