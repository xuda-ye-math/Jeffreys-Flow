clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
addpath(genpath('./potential'));

para = GMM_2D();
fprintf('Generating Final Results Plot for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

lambdas = para.BETA_LIST;
M = length(lambdas) - 1;
fprintf('Detected M=%d steps from BETA_LIST.\n', M);

% =========================================================================
% 2. Visualization Setup
% =========================================================================

% --- Layout Parameters (Numeric values as requested!) ---
% All are relative to figure size (0 to 1)
GAP_W = 0.00;  % Horizontal gap between columns
GAP_H = 0.04;  % Vertical gap between rows
MARG_L = 0.08; % Left margin
MARG_R = 0.02; % Right margin
MARG_T = 0.1; % Top margin
MARG_B = 0.05; % Bottom margin

num_rows = 3;
num_cols = 2;

% Calculate Plot Width/Height
% Total Width = MARG_L + MARG_R + num_cols*plot_w + (num_cols-1)*GAP_W = 1
plot_w = (1 - MARG_L - MARG_R - (num_cols - 1) * GAP_W) / num_cols;
plot_h = (1 - MARG_T - MARG_B - (num_rows - 1) * GAP_H) / num_rows;

% Standard Figure Dimensions
fig_w = 360 * num_cols; 
fig_h = 380 * num_rows;

f = figure('Name', 'Final Results', 'Color', 'w', 'Position', [100, 100, fig_w, fig_h], 'Visible', 'on');

% Color Constants
COLOR_MU = [0.2, 0.4, 0.8];       % Blue (Ref)
COLOR_NU_MALA = [0.85, 0.1, 0.1];      % Red (Gen)
COLOR_NU_NO_MALA = [0.6, 0.3, 0.7];    % Purple

% =========================================================================
% 3. Precompute KDE Background from 'naive_mala'
% =========================================================================
fprintf('Computing KDE Background from Naive MALA...\n');
k = M;
bg_fname = fullfile(data_dir, sprintf('%s_samples_naive_mala_NU_%d.mat', para.ABBR, k));
bg_wname = fullfile(data_dir, sprintf('%s_weights_naive_mala_NU_%d.mat', para.ABBR, k));

bg_samples = []; bg_log_w = [];
if exist(bg_fname, 'file')
    tmp = load(bg_fname); if isfield(tmp,'samples'), bg_samples=tmp.samples; else, vars=fieldnames(tmp); bg_samples=tmp.(vars{1}); end
end
if exist(bg_wname, 'file')
    tmp = load(bg_wname); if isfield(tmp,'weights'), bg_log_w=tmp.weights; else, vars=fieldnames(tmp); bg_log_w=tmp.(vars{1}); end
end

grid_size = 100;
bg_xl = para.X_LIM_PLOT;
bg_yl = para.Y_LIM_PLOT;
[X_bg, Y_bg] = meshgrid(linspace(bg_xl(1), bg_xl(2), grid_size), linspace(bg_yl(1), bg_yl(2), grid_size));
pts_grid = [X_bg(:), Y_bg(:)];
Z_bg = zeros(grid_size, grid_size);

if ~isempty(bg_samples)
    w_norm = normalize_weights(bg_log_w);
    N_bg = size(bg_samples, 1);
    s_pts = min(N_bg, 10000); % Downsample for speed
    if ~isempty(w_norm)
        idx = randsample(N_bg, s_pts, true, w_norm);
    else
        idx = randperm(N_bg, s_pts);
    end
    bg_data = bg_samples(idx, 1:2);
    
    try
        f_vec = ksdensity(bg_data, pts_grid);
        Z_bg = reshape(f_vec, grid_size, grid_size);
    catch e
        fprintf('Background KDE Failed: %s\n', e.message);
    end
end

% =========================================================================
% 4. Plotting ONLY Final Step (k = M)
% =========================================================================
beta_val = lambdas(k + 1);
fprintf('Processing Final Step %d (Beta=%.3f)...\n', k, beta_val);

% Iterate Rows (Top to Bottom logic for Position calculation)
for r = 1 : num_rows
    % Iterate Columns
    for c = 1 : num_cols
        
        % Calculate Position [left, bottom, width, height]
        % Row 1 is Top, Row 3 is Bottom
        % y_pos depends on r:
        % r=1 (Top) -> y = 1 - MARG_T - plot_h
        % r=2       -> y = 1 - MARG_T - plot_h - GAP_H - plot_h
        % General: y = 1 - MARG_T - r*plot_h - (r-1)*GAP_H
        
        pos_x = MARG_L + (c - 1) * (plot_w + GAP_W);
        pos_y = 1 - MARG_T - r * plot_h - (r - 1) * GAP_H;
        
        % Create Axes
        ax = axes('Position', [pos_x, pos_y, plot_w, plot_h]);
        
        % Determine Mode (Column)
        if c == 1
            mode_base = 'naive';
            mode_title = 'Naive';
        else
            mode_base = 'correct';
            mode_title = 'Correct';
        end
        
        % Determine Type (Row) & Filename & Color
        samples = [];
        log_w = [];
        row_title = '';
        scatter_color = COLOR_NU_MALA; 
        
        if r == 1
            % Row 1: reSGLD (MU) -> Blue
            fname = sprintf('%s_samples_%s_MU_%d.mat', para.ABBR, mode_base, k);
            row_title = 'Replica Exchange SGLD';
            scatter_color = COLOR_MU; 
        elseif r == 2
            % Row 2: JF (No MALA) -> Purple
            mode_full = sprintf('%s_no_mala', mode_base);
            fname = sprintf('%s_samples_%s_NU_%d.mat', para.ABBR, mode_full, k);
            wname = sprintf('%s_weights_%s_NU_%d.mat', para.ABBR, mode_full, k);
            row_title = 'Jeffreys Flow (No SVRG)';
            scatter_color = COLOR_NU_NO_MALA;
        elseif r == 3
            % Row 3: JF (MALA) -> Red
            mode_full = sprintf('%s_mala', mode_base);
            fname = sprintf('%s_samples_%s_NU_%d.mat', para.ABBR, mode_full, k);
            wname = sprintf('%s_weights_%s_NU_%d.mat', para.ABBR, mode_full, k);
            row_title = 'Jeffreys Flow (SVRG)';
            scatter_color = COLOR_NU_MALA;
        end
        
        fpath = fullfile(data_dir, fname);
        
        % Load Samples
        if exist(fpath, 'file')
            tmp = load(fpath);
            if isfield(tmp, 'samples')
                samples = tmp.samples;
            else
                vars = fieldnames(tmp);
                samples = tmp.(vars{1});
            end
        end
        
        % Load Weights (if NU)
        if r > 1 && exist(fpath, 'file')
            wpath = fullfile(data_dir, wname);
            if exist(wpath, 'file')
                tmp_w = load(wpath);
                if isfield(tmp_w, 'weights')
                    log_w = tmp_w.weights;
                else
                    vars_w = fieldnames(tmp_w);
                    log_w = tmp_w.(vars_w{1});
                end
            end
        end
        
        % Compute Bias
        bias_val = NaN;
        if ~isempty(samples)
            if isempty(log_w)
                bias_val = compute_bias(para, samples, beta_val, []);
            else
                w_norm = normalize_weights(log_w);
                bias_val = compute_bias(para, samples, beta_val, w_norm);
            end
        end
        
        % Plot Helper
        str_bias = 'N/A';
        if ~isnan(bias_val), str_bias = sprintf('%.4e', bias_val); end
        bias_txt = sprintf('L2 Bias: %s', str_bias);
        
        % Titles
        t_lines = {};
        if r == 1
            t_lines{end+1} = mode_title;
        end
        t_lines{end+1} = bias_txt;
         
        plot_step(ax, samples, scatter_color, para.X_LIM_PLOT, para.Y_LIM_PLOT, t_lines, X_bg, Y_bg, Z_bg);
        
        % Y-Label (Leftmost Column)
        if c == 1
            ylabel(ax, row_title, 'FontWeight', 'bold', 'FontSize', 10);
        end
    end
end

% Manual Global Title (using annotation)
title_txt = sprintf('Jeffreys Flow + reSGLD on 2D GMM');
annotation('textbox', [0, 0.94, 1, 0.05], 'String', title_txt, ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
           'FontSize', 14, 'FontWeight', 'bold');

% Save Figure
save_name = sprintf('%s_plot_NU_results.pdf', para.ABBR);
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

% Show Figure
figure(f);
fprintf('\nFinal plot generated and shown.\n');


% =========================================================================
% Local Functions
% =========================================================================

function w_norm = normalize_weights(log_w)
    if isempty(log_w), w_norm=[]; return; end
    log_w = double(log_w(:)); 
    w = exp(log_w - max(log_w)); 
    w_norm = w / sum(w);
end

function plot_step(ax, samples, col, xl, yl, txt, X_bg, Y_bg, Z_bg)
    hold(ax, 'on'); box(ax, 'on');
    
    % Plot Background KDE (Pure Color, No Contours)
    if nargin >= 9 && ~isempty(Z_bg) && any(Z_bg(:) ~= 0)
        c_lo = [1.0, 1.0, 1.0]; % White (Low Density)
        c_hi = [1.0, 0.85, 0.2]; % Dark Yellow (High Density)
        n_colors = 256;
        cmap = [linspace(c_lo(1),c_hi(1),n_colors)', linspace(c_lo(2),c_hi(2),n_colors)', linspace(c_lo(3),c_hi(3),n_colors)'];
        
        % Use pcolor for smooth continuous shading without contour lines
        pcolor(ax, X_bg, Y_bg, Z_bg);
        shading(ax, 'interp');
        colormap(ax, cmap);
        
        z_max = max(Z_bg(:));
        if z_max == 0, z_max = 1; end
        clim(ax, [0, z_max * 1.05]); % Cap colormap slightly above max for headroom
    end
    
    if ~isempty(samples)
        max_pts = 50000; 
        N = size(samples, 1);
        if N > max_pts
            idx = randperm(N, max_pts);
            samples = samples(idx, :);
        end
        
        scatter(ax, samples(:,1), samples(:,2), 2, col, 'filled', 'MarkerFaceAlpha', 0.2);
    else
        text(ax, 0, 0, 'Missing', 'HorizontalAlignment', 'center', 'Color', 'k');
    end
    
    % Draw Grid
    grid(ax, 'on');
    set(ax, 'Layer', 'top', 'GridAlpha', 0.15);
    
    title(ax, txt, 'Interpreter', 'none', 'FontSize', 10); 
    set(ax, 'XTick', [], 'YTick', [], 'LineWidth', 0.8);
    xlim(ax, xl); ylim(ax, yl);
    axis(ax, 'square');
end