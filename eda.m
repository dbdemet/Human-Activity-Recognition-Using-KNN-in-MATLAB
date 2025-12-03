function eda(dataDir, outDir)
%EDA Exploratory data analysis for UCI HAR dataset.
%   eda(dataDir) creates basic plots (class distribution, sample signals,
%   spectrogram) and saves them into outDir (optional). If outDir is not
%   provided, uses './results_figs'. Existing files are not overwritten; a
%   run-specific subfolder will be created when appropriate.

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(pwd,'UCI HAR Dataset');
end
if nargin < 2 || isempty(outDir), outDir = fullfile(pwd,'results_figs'); end

% ensure outDir exists and avoid clobbering existing files by using a run subfolder
if exist(outDir,'dir')==7 && ~isempty(dir(fullfile(outDir,'*')))
    [~,last] = fileparts(outDir);
    if ~startsWith(last,'run_')
        ts = datestr(now,'yyyymmdd_HHMMSS');
        outDir = fullfile(outDir, ['run_' ts]);
        if ~exist(outDir,'dir'), mkdir(outDir); end
    end
else
    if ~exist(outDir,'dir'), mkdir(outDir); end
end

sets = {'train','test'};
signals = {'body_acc_x','body_acc_y','body_acc_z', 'body_gyro_x','body_gyro_y','body_gyro_z'};

% load labels
labels = [];
for t = 1:2
    yfile = fullfile(dataDir,sets{t},['y_',sets{t},'.txt']);
    labels = [labels; readmatrix(yfile)];
end

% outDir is already prepared above

% Class distribution
fig1 = figure('Visible','off','Color','white','InvertHardcopy','off');
classes = unique(labels);
counts = arrayfun(@(c) sum(labels==c), classes);
bar(classes, counts);
xlabel('Class'); ylabel('Count'); title('Class distribution (train+test)');
    savefig(fig1, fullfile(outDir,'eda_class_distribution.fig'));
    try, exportgraphics(fig1, fullfile(outDir,'eda_class_distribution.png')); end
close(fig1);

% Plot a few sample time-series from Inertial Signals
% We'll take one example from middle of dataset
idxSample = round(size(labels,1)/2);
L = 128; fs = 50;
fig2 = figure('Visible','off','Color','white','InvertHardcopy','off');
for s = 1:3
    fname = fullfile(dataDir,'train','Inertial Signals',[signals{s},'_train.txt']);
    if ~isfile(fname), fname = fullfile(dataDir,'test','Inertial Signals',[signals{s},'_test.txt']); end
    M = readmatrix(fname);
    if size(M,1) < idxSample, idx = 1; else idx = idxSample; end
    x = M(idx,:);
    subplot(3,1,s);
    plot((0:L-1)/fs, x);
    ylabel(signals{s});
    if s==1, title(sprintf('Sample %d time-series', idxSample)); end
    if s==3, xlabel('Time (s)'); end
end
savefig(fig2, fullfile(outDir,'eda_sample_timeseries.fig'));
try, exportgraphics(fig2, fullfile(outDir,'eda_sample_timeseries.png')); end
close(fig2);

% Spectrogram for one signal
sigFile = fullfile(dataDir,'train','Inertial Signals','body_acc_x_train.txt');
if isfile(sigFile)
    M = readmatrix(sigFile);
    x = M(idxSample,:);
    fig3 = figure('Visible','off','Color','white','InvertHardcopy','off');
    % Use spectrogram if available, otherwise use a simple STFT implementation
    if exist('spectrogram','file') == 2
        spectrogram(x, 64, 60, 256, fs, 'yaxis');
    else
        % simple STFT and plot (toolbox-free)
        win = 64; noverlap = 60; nfft = 256;
        step = win - noverlap;
        w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1));
        idxs = 1:step:(length(x)-win+1);
        Smat = zeros(nfft/2+1, length(idxs));
        for k = 1:length(idxs)
            seg = x(idxs(k):idxs(k)+win-1) .* w';
            X = fft(seg, nfft);
            Smat(:,k) = abs(X(1:nfft/2+1));
        end
        t = (idxs-1)/fs;
        f = (0:nfft/2)*(fs/nfft);
        imagesc(t, f, 20*log10(Smat)); axis xy; colormap(parula);
        % Ensure axes background is white
        ax = gca; ax.Color = 'white';
        xlabel('Time (s)'); ylabel('Frequency (Hz)');
    end
    title(sprintf('Spectrogram - sample %d (body_acc_x)', idxSample));
    savefig(fig3, fullfile(outDir,'eda_spectrogram.fig'));
    try, exportgraphics(fig3, fullfile(outDir,'eda_spectrogram.png')); end
    close(fig3);
end

fprintf('EDA complete. Figures saved to %s\n', outDir);
end
