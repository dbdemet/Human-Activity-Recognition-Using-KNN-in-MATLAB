function varargout = spectrogram_utils(dataDir, setType, sampleIdx, mode, outFile)
%SPECTROGRAM_UTILS Unified spectrogram utilities: feature extraction and visualization
%   feat = spectrogram_utils(dataDir, setType, sampleIdx, 'features')
%   spectrogram_utils(dataDir, setType, sampleIdx, 'save', outFile)
%
%   Modes:
%     'features' - Extract band-power features from spectrogram
%     'save'     - Save spectrogram visualization to file
%
%   Examples:
%     feat = spectrogram_utils('UCI HAR Dataset', 'train', 1, 'features');
%     spectrogram_utils('UCI HAR Dataset', 'train', 50, 'save', 'spect.png');

if nargin < 1 || isempty(dataDir), dataDir = fullfile(pwd,'UCI HAR Dataset'); end
if nargin < 2 || isempty(setType), setType = 'train'; end
if nargin < 3 || isempty(sampleIdx), sampleIdx = 1; end
if nargin < 4 || isempty(mode), mode = 'features'; end

fs = 50;
% Find body_acc_x file
fname = fullfile(dataDir, setType, 'Inertial Signals', ['body_acc_x_' setType '.txt']);
if ~isfile(fname)
    fname = fullfile(dataDir, 'train', 'Inertial Signals', 'body_acc_x_train.txt');
    if ~isfile(fname)
        fname = fullfile(dataDir, 'test', 'Inertial Signals', 'body_acc_x_test.txt');
    end
end
if ~isfile(fname)
    error('body_acc_x file not found for %s', setType);
end

M = readmatrix(fname);
if sampleIdx < 1 || sampleIdx > size(M,1), error('sampleIdx out of range'); end
x = double(M(sampleIdx,:));

switch lower(mode)
    case 'features'
        % Extract band-power features
        N = length(x); nfft = 256; 
        Xf = abs(fft(x, nfft)).^2; 
        P = Xf(1:floor(nfft/2)+1);
        f = (0:floor(nfft/2))*(fs/nfft);
        % define bands and compute power
        bands = [0.1 3; 3 12; 12 20; 20 40];
        feat = zeros(1, size(bands,1));
        for b = 1:size(bands,1)
            idx = f >= bands(b,1) & f <= bands(b,2);
            feat(b) = sum(P(idx));
        end
        varargout{1} = feat;
        
    case 'save'
        % Save spectrogram visualization
        if nargin < 5 || isempty(outFile)
            outFile = fullfile(pwd, 'results_figs', sprintf('sample_spectrogram_%d.png', sampleIdx));
        end
        
        if ~exist(fileparts(outFile), 'dir'), mkdir(fileparts(outFile)); end
        fig = figure('Visible', 'off', 'Color', 'white', 'InvertHardcopy', 'off');
        
        if exist('spectrogram', 'file') == 2
            spectrogram(x, 64, 60, 256, fs, 'yaxis');
        else
            % simple STFT fallback (toolbox-free)
            win = 64; noverlap = 60; nfft = 256;
            step = win - noverlap;
            w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1)); % Hamming window
            idxs = 1:step:(length(x)-win+1);
            Smat = zeros(nfft/2+1, length(idxs));
            for k = 1:length(idxs)
                seg = x(idxs(k):idxs(k)+win-1) .* w';
                Xf = fft(seg, nfft);
                Smat(:,k) = abs(Xf(1:nfft/2+1));
            end
            t = (idxs-1)/fs;
            f = (0:nfft/2)*(fs/nfft);
            imagesc(t, f, 20*log10(Smat)); axis xy; colormap(parula);
            ax = gca; ax.Color = 'white';
            xlabel('Time (s)'); ylabel('Frequency (Hz)');
        end
        title(sprintf('Spectrogram sample %d', sampleIdx));
        try
            exportgraphics(fig, outFile);
        catch
            saveas(fig, outFile);
        end
        close(fig);
        fprintf('Spectrogram saved to %s\n', outFile);
        if nargout > 0, varargout{1} = outFile; end
        
    otherwise
        error('Unknown mode: %s. Use ''features'' or ''save''', mode);
end
end



