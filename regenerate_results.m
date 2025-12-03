function regenerate_results(matFile)
% REGENERATE_RESULTS Recreate presentation-ready EDA & results images
% regenerate_results()                  - loads default MAT results candidates
% regenerate_results('my_results.mat')  - load a specific MAT file
%
% Writes PNGs to `results_figs` in the project root:
% - class_distribution.png
% - sample_time_series.png
% - sample_spectrogram.png
% - knn_confusion_matrix.png
%
% This file is self-contained and includes helper functions below.

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'results_figs');
if ~exist(outDir,'dir'), mkdir(outDir); end

% Clean results_figs directory completely before generating new outputs
fprintf('Cleaning results_figs directory...\n');
oldFiles = dir(fullfile(outDir,'*'));
for i=1:numel(oldFiles)
    if ~oldFiles(i).isdir && ~startsWith(oldFiles(i).name,'.')
        try delete(fullfile(outDir,oldFiles(i).name)); catch, end
    end
end
fprintf('Cleaned results_figs directory.\n');

% Load MAT results if provided or available
S = [];
if nargin>=1 && ~isempty(matFile)
    f = fullfile(root,matFile);
    if exist(f,'file')==2
        try S = load(f); fprintf('Loaded results from %s\n', f); catch ME, warning('Could not load %s: %s', f, ME.message); end
    else
        fprintf('Requested MAT file %s not found; will try default candidates.\n', matFile);
    end
end
if isempty(S)
    candidates = {'har_models_results.mat','training_results_summary.mat','results.mat'};
    for k=1:numel(candidates)
        f = fullfile(root,candidates{k});
        if exist(f,'file')==2
            try S = load(f); fprintf('Loaded results from %s\n', f); break; catch, end
        end
    end
end

% Locate dataset if available
datasetDir = locate_ucihar(root);

% (Directory already cleaned at the beginning)

% Try to extract a results struct from MAT file
results = [];
if ~isempty(S)
    fn = fieldnames(S);
    for k=1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && (isfield(v,'names') || isfield(v,'metrics'))
            results = v; break;
        end
    end
    if isempty(results) && isfield(S,'results'), results = S.results; end
end

% Default class names
classNames = {'WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING'};
if ~isempty(results) && isfield(results,'class_names')
    try classNames = results.class_names; end
end

% 1) Class distribution (from dataset)
if ~isempty(datasetDir)
    try
        ytrain = load(fullfile(datasetDir,'train','y_train.txt'));
        ytest  = load(fullfile(datasetDir,'test','y_test.txt'));
        yall = [ytrain; ytest];
        counts_all = arrayfun(@(k) sum(yall==k), 1:numel(classNames));
        fig = figure('Visible','off','Color','w','Position',[200 200 900 600]);
        b = bar(counts_all,'FaceColor','flat'); b.CData = repmat([12 82 160]/255,numel(counts_all),1);
        set(gca,'XTick',1:numel(classNames),'XTickLabel',classNames,'XTickLabelRotation',30,'FontSize',12);
        ylabel('Number of samples'); title('Class distribution (train+test)','FontSize',16);
        out = fullfile(outDir,'class_distribution.png'); exportgraphics(fig,out,'Resolution',300); savefig(fig,fullfile(outDir,'class_distribution.fig')); close(fig);
        fprintf('Wrote %s\n', out);
    catch ME
        fprintf('Class distribution generation failed: %s\n', ME.message);
    end
else
    fprintf('UCI HAR Dataset not found; skipping class distribution.\n');
end

% 2) Sample time series + spectrogram (canonical sample 3676)
sampleIndex = 3676;
try
    if isempty(datasetDir)
        error('Dataset not available to extract sample %d', sampleIndex);
    end
    % determine train/test split
    bx_train_file = fullfile(datasetDir,'train','Inertial Signals','body_acc_x_train.txt');
    if exist(bx_train_file,'file')==2
        bx_train = load(bx_train_file); nTrain = size(bx_train,1);
    else
        nTrain = 0;
    end
    if sampleIndex <= nTrain && nTrain>0
        setName = 'train'; localIdx = sampleIndex;
    else
        setName = 'test'; localIdx = sampleIndex - nTrain;
    end
    sigFolder = fullfile(datasetDir,setName,'Inertial Signals');
    fx = fullfile(sigFolder,sprintf('body_acc_x_%s.txt',setName));
    fy = fullfile(sigFolder,sprintf('body_acc_y_%s.txt',setName));
    fz = fullfile(sigFolder,sprintf('body_acc_z_%s.txt',setName));
    if ~(exist(fx,'file')==2 && exist(fy,'file')==2 && exist(fz,'file')==2)
        error('Inertial signal files missing in %s', sigFolder);
    end
    bx = load(fx); by = load(fy); bz = load(fz);
    if localIdx<1 || localIdx>size(bx,1), error('Sample index out of range'); end
    x = double(bx(localIdx,:)); y = double(by(localIdx,:)); z = double(bz(localIdx,:));
    t = (0:numel(x)-1)/50;
    fig = figure('Visible','off','Color','w','Position',[150 150 1000 500]);
    plot(t,x,'-','LineWidth',1.2); hold on; plot(t,y,'-','LineWidth',1.2); plot(t,z,'-','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Acceleration'); legend({'body\_acc\_x','body\_acc\_y','body\_acc\_z'},'Location','best');
    title(sprintf('Sample time series %d (%s)', sampleIndex, setName),'FontSize',14); outts = fullfile(outDir,'sample_time_series.png'); exportgraphics(fig,outts,'Resolution',300); savefig(fig,fullfile(outDir,'sample_time_series.fig')); close(fig);
    fprintf('Wrote %s\n', outts);

    % Spectrogram matching the reference image format
    fs = 50; % sampling rate for UCI HAR
    if ~isempty(which('spectrogram'))
        % Parameters to match reference: window=64, overlap=60, nfft=256
        win = 64; noverlap = 60; nfft = 256;
        if ~isempty(which('hamming'))
            w = hamming(win);
        else
            w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1));
        end
        try
            [S,F,T] = spectrogram(x, w, noverlap, nfft, fs);
            % Validate outputs
            if isempty(S) || isempty(F) || isempty(T) || any(~isfinite(F)) || any(~isfinite(T))
                error('spectrogram produced empty or non-finite outputs, falling back to manual STFT');
            end
            % Convert to power spectral density (magnitude squared)
            P = abs(S).^2;
            % Convert to dB scale
            Pdb = 10*log10(P + eps);
            % Light 2D Gaussian smoothing (separable) to reduce speckle/noise
            try
                gsize = 7; sigma = 1.0;
                xg = -(floor(gsize/2)) : floor(gsize/2);
                g = exp(-(xg.^2)/(2*sigma^2)); g = g / sum(g);
                Pdb = conv2(g, g, Pdb, 'same');
            catch
                % if conv2 not available for some reason, continue without smoothing
            end

            % Ensure frequency axis is increasing
            if F(1) > F(end)
                F = flipud(F(:)); Pdb = flipud(Pdb);
            end

            % Robust color limits using percentiles to suppress outliers and emphasize warm bands
            vs = sort(Pdb(:));
            idxTop = max(1, round(0.99 * numel(vs)));
            top = vs(idxTop);
            bottom = top - 60; % 60 dB range emphasizes orange/yellow
            clim = [bottom top];

            % High-resolution figure for presentation using pcolor + interpolated shading
            fig = figure('Visible','off','Color','w','InvertHardcopy','off','Position',[40 40 2200 1400],'Renderer','opengl');
            % Prepare smoothed spectrogram matrix and meshgrid for pcolor
            S_smooth = Pdb; % already smoothed above when possible
            vmin = -90; vmax = -20;
            [TT,FF] = meshgrid(T,F);
            h = pcolor(TT,FF,S_smooth); set(h,'EdgeColor','none'); shading interp; % shading interp ~ 'gouraud'
            set(gca,'YDir','normal');
            % Try turbo colormap, fallback to yellow_spectro
            try
                cmap = turbo(512); colormap(cmap);
            catch
                colormap(yellow_spectro(512));
            end
            hb = colorbar('EastOutside'); hb.FontSize = 14; hb.Label.String = 'Power (dB)';
            % Apply explicit dynamic range to emphasize warm bands
            try, caxis([vmin vmax]); catch, end
            % Frequency ticks at 1 Hz intervals
            try, yticks(0:1:25); catch, end
            xlabel('Time (s)','FontSize',16);
            ylabel('Frequency (Hz)','FontSize',16);
            title(sprintf('Spectrogram sample %d', sampleIndex),'FontSize',22,'FontWeight','bold');
            set(gca,'FontSize',14);

            outspec = fullfile(outDir,'sample_spectrogram.png');
            exportgraphics(fig,outspec,'Resolution',600);
            savefig(fig,fullfile(outDir,'sample_spectrogram.fig'));
            fprintf('Wrote %s\n', outspec);
            close(fig);

            % Also save a log-normalized magnitude view (sharper small peaks)
            try
                Sabs = abs(S); Sabs(Sabs<=0) = eps;
                fig2 = figure('Visible','off','Color','w','Position',[40 40 2200 1400],'Renderer','opengl');
                [TT2,FF2] = meshgrid(T,F);
                h2 = pcolor(TT2,FF2,Sabs); set(h2,'EdgeColor','none'); shading interp;
                try, cmap = turbo(512); colormap(cmap); catch, colormap(yellow_spectro(512)); end
                ax2 = gca; ax2.ColorScale = 'log';
                % Use log-normalization vmin/vmax similar to LogNorm
                try, caxis([1e-4 max(Sabs(:))]); catch, end
                try, yticks(0:1:25); catch, end
                hb2 = colorbar('EastOutside'); hb2.FontSize = 14; hb2.Label.String = 'Magnitude (log scale)';
                xlabel('Time (s)','FontSize',16); ylabel('Frequency (Hz)','FontSize',16);
                title(sprintf('Spectrogram (log mag) sample %d', sampleIndex),'FontSize',22,'FontWeight','bold'); set(gca,'FontSize',14);
                outspec2 = fullfile(outDir,'sample_spectrogram_log.png'); exportgraphics(fig2,outspec2,'Resolution',600); savefig(fig2,fullfile(outDir,'sample_spectrogram_log.fig')); close(fig2);
                fprintf('Wrote %s\n', outspec2);
            catch LGE
                fprintf('Log-normalized spectrogram save failed: %s\n', LGE.message);
            end
        catch SPE
            fprintf('Toolbox spectrogram failed (%s), will attempt manual STFT fallback.\n', SPE.message);
            % fall through to manual STFT section below
            % Use same parameters for manual STFT
            win = 64; noverlap = 60; nfft = 256; hop = win - noverlap;
            L = numel(x); starts = 1:hop:(L-win+1); if isempty(starts), starts=1; end
            Smat = zeros(nfft/2+1,numel(starts));
            w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1));
            for k=1:numel(starts)
                seg = x(starts(k):starts(k)+win-1).*w'; Xf = fft(seg,nfft); Smat(:,k) = abs(Xf(1:nfft/2+1));
            end
            tvec = (starts+win/2-1)/fs; fvec = (0:(nfft/2))*(fs/nfft);
            P = Smat.^2; Pdb = 10*log10(P + eps);
            % Light smoothing to reduce speckle and emphasize coherent bands
            try
                gsize = 7; sigma = 1.0;
                xg = -(floor(gsize/2)) : floor(gsize/2);
                g = exp(-(xg.^2)/(2*sigma^2)); g = g / sum(g);
                Pdb = conv2(g, g, Pdb, 'same');
            catch
            end
            if fvec(1) > fvec(end)
                fvec = flipud(fvec(:)); Pdb = flipud(Pdb);
            end
            % Robust color limits
            vs = sort(Pdb(:)); idxTop = max(1, round(0.99 * numel(vs))); top = vs(idxTop); bottom = top - 60; clim = [bottom top];
            fig = figure('Visible','off','Color','w','InvertHardcopy','off','Position',[40 40 2200 1400],'Renderer','opengl');
            % Prepare smoothed spectrogram matrix and meshgrid for pcolor
            S_smooth = Pdb;
            vmin = -90; vmax = -20;
            [TT,FF] = meshgrid(tvec,fvec);
            h = pcolor(TT,FF,S_smooth); set(h,'EdgeColor','none'); shading interp;
            set(gca,'YDir','normal');
            try, cmap = turbo(512); colormap(cmap); catch, colormap(yellow_spectro(512)); end
            hb = colorbar('EastOutside'); hb.FontSize = 14; hb.Label.String = 'Power (dB)';
            try, caxis([vmin vmax]); end
            try, yticks(0:1:25); catch, end
            try, ylim([0 25]); catch, end
            xlabel('Time (s)','FontSize',16); ylabel('Frequency (Hz)','FontSize',16);
            title(sprintf('Spectrogram sample %d (fallback)', sampleIndex),'FontSize',22,'FontWeight','bold'); set(gca,'FontSize',14);
            outspec = fullfile(outDir,'sample_spectrogram.png'); exportgraphics(fig,outspec,'Resolution',600); savefig(fig,fullfile(outDir,'sample_spectrogram.fig'));
            fprintf('Wrote %s\n', outspec); close(fig);

            % Log-normalized magnitude view for fallback path
            try
                Sabs = Smat; Sabs(Sabs<=0) = eps;
                fig2 = figure('Visible','off','Color','w','Position',[40 40 2200 1400],'Renderer','opengl');
                [TT2,FF2] = meshgrid(tvec,fvec);
                h2 = pcolor(TT2,FF2,Sabs); set(h2,'EdgeColor','none'); shading interp;
                try, cmap = turbo(512); colormap(cmap); catch, colormap(yellow_spectro(512)); end
                ax2 = gca; ax2.ColorScale = 'log';
                try, caxis([1e-4 max(Sabs(:))]); catch, end
                try, yticks(0:1:25); catch, end
                hb2 = colorbar('EastOutside'); hb2.FontSize = 14; hb2.Label.String = 'Magnitude (log scale)';
                xlabel('Time (s)','FontSize',16); ylabel('Frequency (Hz)','FontSize',16);
                title(sprintf('Spectrogram (log mag) sample %d', sampleIndex),'FontSize',22,'FontWeight','bold'); set(gca,'FontSize',14);
                outspec2 = fullfile(outDir,'sample_spectrogram_log.png'); exportgraphics(fig2,outspec2,'Resolution',600); savefig(fig2,fullfile(outDir,'sample_spectrogram_log.fig')); close(fig2);
                fprintf('Wrote %s\n', outspec2);
            catch LG2
                fprintf('Log-normalized fallback spectrogram save failed: %s\n', LG2.message);
            end
        end
    else
        % Manual STFT fallback (matching reference parameters: win=64, noverlap=60, nfft=256)
        win = 64; noverlap = 60; nfft = 256; hop = win - noverlap;
        L = numel(x); starts = 1:hop:(L-win+1); if isempty(starts), starts=1; end
        Smat = zeros(nfft/2+1,numel(starts));
        w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1));
        for k=1:numel(starts)
            seg = x(starts(k):starts(k)+win-1).*w'; Xf = fft(seg,nfft); Smat(:,k) = abs(Xf(1:nfft/2+1));
        end
        tvec = (starts+win/2-1)/fs; fvec = (0:(nfft/2))*(fs/nfft);
        % Convert to power and dB scale
        P = Smat.^2;
        Pdb = 10*log10(P + eps);
        % Light smoothing to reduce speckle and emphasize coherent bands
        try
            gsize = 7; sigma = 1.0;
            xg = -(floor(gsize/2)) : floor(gsize/2);
            g = exp(-(xg.^2)/(2*sigma^2)); g = g / sum(g);
            Pdb = conv2(g, g, Pdb, 'same');
        catch
        end

        % Robust color limits using percentiles
        vs = sort(Pdb(:)); idxTop = max(1, round(0.99 * numel(vs))); top = vs(idxTop); bottom = top - 60; clim = [bottom top];

        % Use pcolor with smoothing to create a high-contrast, warm spectrogram
        S_smooth = Pdb;
        vmin = clim(1); vmax = clim(2);
        fig = figure('Visible','off','Color','w','InvertHardcopy','off','Position',[120 80 1600 1000],'Renderer','opengl');
        [TT,FF] = meshgrid(tvec,fvec);
        h = pcolor(TT,FF,S_smooth); set(h,'EdgeColor','none'); shading interp;
        set(gca,'YDir','normal');
        try, cmap = turbo(512); colormap(cmap); catch, colormap(yellow_spectro(512)); end
        hb = colorbar; hb.FontSize = 12; hb.Label.String = 'Power (dB)';
        try, caxis([vmin vmax]); end
        try, yticks(0:1:25); catch, end
        ylim([0 25]);
        xlabel('Time (s)','FontSize',14);
        ylabel('Frequency (Hz)','FontSize',14);
        title(sprintf('Spectrogram sample %d', sampleIndex),'FontSize',18,'FontWeight','bold');
        set(gca,'FontSize',12);
        outspec = fullfile(outDir,'sample_spectrogram.png');
        exportgraphics(fig,outspec,'Resolution',600);
        savefig(fig,fullfile(outDir,'sample_spectrogram.fig'));
        fprintf('Wrote %s\n', outspec);
        close(fig);
    end
catch ME
    fprintf('Sample time series / spectrogram generation failed: %s\n', ME.message);
end

% -------------------------------------------------------------------------
% Regenerate spectrogram in the original simple MATLAB style so that
% sample_spectrogram.png matches the earlier reference image exactly.
% This uses the same parameters and plotting code as archive/save_spectrogram_sample.m
% (warm parula colormap, blocky look, no colorbar).
% -------------------------------------------------------------------------
try
    fs = 50;
    figRef = figure('Visible','off','Color','white','InvertHardcopy','off', ...
                    'Position',[100 100 1140 768]);
    if exist('spectrogram','file') == 2
        spectrogram(x, 64, 60, 256, fs, 'yaxis');
    else
        % toolbox yoksa, basit STFT ile yaklaşık aynı görüntüyü üret
        win = 64; noverlap = 60; nfft = 256;
        step = win - noverlap;
        w = 0.54 - 0.46*cos(2*pi*(0:win-1)'/(win-1));
        idxs = 1:step:(length(x)-win+1);
        Smat = zeros(nfft/2+1, length(idxs));
        for k = 1:length(idxs)
            seg = x(idxs(k):idxs(k)+win-1) .* w';
            Xf = fft(seg, nfft);
            Smat(:,k) = abs(Xf(1:nfft/2+1));
        end
        tLoc = (idxs-1)/fs;
        fLoc = (0:nfft/2)*(fs/nfft);
        imagesc(tLoc, fLoc, 20*log10(Smat+eps)); axis xy;
        colormap(parula);
        xlabel('Time (s)'); ylabel('Frequency (Hz)');
    end
    title(sprintf('Spectrogram sample %d', sampleIndex));
    % Orijinal görselde colorbar yok, o yüzden varsa kaldır
    cb = colorbar('peer',gca);
    if ~isempty(cb) && isvalid(cb)
        delete(cb);
    end

    outspecRef = fullfile(outDir,'sample_spectrogram.png');
    exportgraphics(figRef, outspecRef, 'Resolution',300);
    savefig(figRef, fullfile(outDir,'sample_spectrogram.fig'));
    close(figRef);
    fprintf('Overwrote spectrogram with reference-style version at %s\n', outspecRef);
catch MEref
    fprintf('Reference-style spectrogram regeneration failed: %s\n', MEref.message);
end

% 3) kNN confusion matrix (if available in MAT results)
if ~isempty(results)
    try
        C = [];
        if isfield(results,'metrics')
            for m=1:numel(results.metrics)
                met = results.metrics{m};
                if isstruct(met) && isfield(met,'confusion') && ~isempty(met.confusion)
                    C = met.confusion; break;
                end
            end
        end
        if isempty(C) && isfield(results,'confusion')
            C = results.confusion;
        end
        if ~isempty(C)
            % Blue colormap matching reference image (light -> deep blue)
            blues = interp1([0 1],[1 1 1; 0.06 0.25 0.54],linspace(0,1,256));
            fig = figure('Visible','off','Color','w','Position',[80 80 1400 920]);
            imagesc(C); colormap(blues);
            ax = gca;
            % Ticks and labels
            ax.XTick = 1:numel(classNames);
            ax.XTickLabel = classNames;
            ax.XTickLabelRotation = 30;
            ax.YTick = 1:numel(classNames);
            ax.YTickLabel = classNames;
            ax.FontSize = 12;
            ax.TickLength = [0 0];

            % Determine model name for title (try to find kNN)
            modelName = 'kNN';
            if ~isempty(results) && isfield(results,'names')
                for m=1:numel(results.names)
                    nameStr = results.names{m};
                    if ischar(nameStr) && ~isempty(regexpi(nameStr,'knn|k[- ]?nn|knearest','once'))
                        modelName = nameStr;
                        break;
                    end
                end
            end
            
            % Title and axis labels in blue, centered top like the example
            title(sprintf('%s - Confusion matrix',modelName),'FontSize',20,'FontWeight','bold','Color',[0 0.3 0.6]);
            xlabel('Predicted','FontSize',14,'Color',[0 0.3 0.6]); ylabel('True','FontSize',14,'Color',[0 0.3 0.6]);

            % Colorbar to the right with clean ticks
            hb = colorbar('EastOutside'); hb.FontSize = 12; hb.TickDirection = 'out';

            % Draw white grid lines between cells (like the example)
            hold on;
            [nr,nc] = size(C);
            for r=0:nr
                plot([0.5 nc+0.5],[r+0.5 r+0.5],'Color',[1 1 1],'LineWidth',1.2);
            end
            for c=0:nc
                plot([c+0.5 c+0.5],[0.5 nr+0.5],'Color',[1 1 1],'LineWidth',1.2);
            end

            % Overlay numeric labels in each cell (matching reference format)
            mx = max(C(:));
            for irow=1:nr
                for jcol=1:nc
                    val = C(irow,jcol);
                    if mx>50 && abs(round(val)-val)<eps % integer counts
                        s = sprintf('%d',round(val));
                    else
                        s = sprintf('%.2f',val);
                    end
                    % choose text color: white on dark cells, deep blue otherwise
                    txtColor = [0 0.2 0.45];
                    if val > mx*0.6, txtColor = [1 1 1]; end
                    text(jcol,irow,s,'HorizontalAlignment','center','VerticalAlignment','middle','FontWeight','bold','FontSize',14,'Color',txtColor);
                end
            end

            axis tight; axis ij;
            outpng = fullfile(outDir,'knn_confusion_matrix.png');
            exportgraphics(fig,outpng,'Resolution',300);
            savefig(fig,fullfile(outDir,'knn_confusion_matrix.fig'));
            close(fig);
            fprintf('Wrote %s\n', outpng);
        else
            fprintf('No confusion matrix found in MAT results; skipping kNN confusion output.\n');
        end
    catch ME
        fprintf('Confusion matrix generation failed: %s\n', ME.message);
    end
else
    fprintf('No MAT results found; skipping confusion matrix.\n');
end

fprintf('Finished — images (if generated) are in: %s\n', outDir);
end

%% Helper functions (local, explicit end for clarity)
function cmap = sky_colormap(n)
if nargin<1, n=256; end
stops = [1 1 1; 0.93 0.96 0.99; 0.8 0.9 0.97; 0.55 0.75 0.95; 0.25 0.5 0.85; 0.1 0.2 0.6];
x = linspace(0,1,size(stops,1)); xi = linspace(0,1,n)'; cmap = interp1(x,stops,xi); cmap = max(0,min(1,cmap));
end

function p = locate_ucihar(root)
% Try common names and a limited sibling/parent search
p = '';
cands = {fullfile(root,'UCI HAR Dataset'), fullfile(root,'UCI_HAR_Dataset'), fullfile(root,'dataset','UCI HAR Dataset')};
for i=1:numel(cands), if exist(cands{i},'dir')==7, p = cands{i}; return; end, end
parent = fileparts(root);
searchDirs = {root,parent};
for dI=1:numel(searchDirs)
    droot = searchDirs{dI}; if isempty(droot), continue; end
    listing = dir(droot);
    for j=1:numel(listing)
        if listing(j).isdir && ~startsWith(listing(j).name,'.')
            lname = lower(listing(j).name);
            if contains(lname,'uci') && contains(lname,'har'), p = fullfile(droot,listing(j).name); return; end
        end
    end
end
end

function cmap = yellow_spectro(n)
% YELLOW_SPECTRO Create a warm yellow→orange→green→blue colormap for spectrograms
% Enhanced to provide stronger orange/yellow emphasis for clearer warm bands
if nargin<1, n=256; end
% Color stops: dark blue -> blue -> cyan -> green -> yellow -> strong orange
stops = [
    0.02 0.12 0.40;  % deep blue (lowest power)
    0.12 0.55 0.88;  % blue
    0.22 0.82 0.80;  % cyan
    0.65 0.85 0.35;  % green
    1.00 0.92 0.20;  % bright yellow
    1.00 0.55 0.05   % strong orange (highest power)
];
x = linspace(0,1,size(stops,1)); 
xi = linspace(0,1,n)'; 
cmap = interp1(x,stops,xi,'linear'); 
cmap = max(0,min(1,cmap));
end