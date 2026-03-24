clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
addpath(genpath('./data'));

para_path = './data/para.mat';
if ~exist(para_path, 'file')
    error('Configurations para.mat not found. Please run parameters.m first.');
end
load(para_path); % Loads 'para' struct

fprintf('Generating 2x2 Marginals Plot...\\n');

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

RESOLUTION_DPI = 600; % Resolution interface for exporting plots

M = para.M;
k = M; % We plot the final stage results

% =========================================================================
% 2. Visualization Setup
% =========================================================================

% --- Layout Parameters ---
GAP_W = 0.05;  % Horizontal gap between columns
GAP_H = 0.08;  % Vertical gap between rows
MARG_L = 0.08; % Left margin
MARG_R = 0.04; % Right margin
MARG_T = 0.08; % Top margin
MARG_B = 0.08; % Bottom margin

num_rows = 2;
num_cols = 2;

% Calculate Plot Width/Height
plot_w = (1 - MARG_L - MARG_R - (num_cols - 1) * GAP_W) / num_cols;
plot_h = (1 - MARG_T - MARG_B - (num_rows - 1) * GAP_H) / num_rows;

% Standard Figure Dimensions (Square aspect ratio for 2D map)
fig_w = 400 * num_cols; 
fig_h = 400 * num_rows;

f = figure('Name', 'Final Results - Theta_1 2D Scatter', 'Color', 'w', 'Position', [100, 100, fig_w, fig_h], 'Visible', 'on');

% Color Constants (Matching User Request and 2D GMM)
COLOR_MU = [0.2, 0.4, 0.8];       % Blue (reSGLD)
COLOR_NU_E = [0.85, 0.1, 0.1];    % Red (Exact)
COLOR_NU_F = [0.85, 0.4, 0.1];    % Orange (Full)
COLOR_NU_S = [0.9, 0.75, 0.1];    % Yellow (Stochastic)

TRUE_COLOR = [0.1, 0.8, 0.3];     % Green for Ground Truth Line

% Subplot Configurations
% 1: MU (Top-Left)
% 2: NU_e (Top-Right)
% 3: NU_f (Bottom-Left)
% 4: NU_s (Bottom-Right)

plot_configs = struct();
plot_configs(1).name = sprintf('samples_MU_%d.mat', k);
plot_configs(1).title = 'Replica Exchange SGLD';
plot_configs(1).color = COLOR_MU;
plot_configs(1).row = 1;
plot_configs(1).col = 1;
plot_configs(1).weighted = false;

plot_configs(2).name = sprintf('samples_e_NU_%d.mat', k);
plot_configs(2).wname = sprintf('weights_e_NU_%d.mat', k);
plot_configs(2).title = 'Jeffreys Flow: Exact';
plot_configs(2).color = COLOR_NU_E;
plot_configs(2).row = 1;
plot_configs(2).col = 2;
plot_configs(2).weighted = true;

plot_configs(3).name = sprintf('samples_f_NU_%d.mat', k);
plot_configs(3).wname = sprintf('weights_f_NU_%d.mat', k);
plot_configs(3).title = 'Jeffreys Flow: Full Potential';
plot_configs(3).color = COLOR_NU_F;
plot_configs(3).row = 2;
plot_configs(3).col = 1;
plot_configs(3).weighted = true;

plot_configs(4).name = sprintf('samples_s_NU_%d.mat', k);
plot_configs(4).wname = sprintf('weights_s_NU_%d.mat', k);
plot_configs(4).title = 'Jeffreys Flow: Stochastic';
plot_configs(4).color = COLOR_NU_S;
plot_configs(4).row = 2;
plot_configs(4).col = 2;
plot_configs(4).weighted = true;

% Ground truth value for Theta_1 (Assuming para.theta_true has 8 dimensions for 4 sources)
theta_true_x = para.theta_true(1:2:end);
theta_true_y = para.theta_true(2:2:end);

% =========================================================================
% 3. Pre-computing Densities for Global Colorbar
% =========================================================================
fprintf('Pre-computing 2D KDE densities for global colormap...\n');

% We need to find the global maximum density across all 4 subplots
global_max_density = 0;
all_f_dens = cell(4, 1);
all_X = cell(4, 1);
all_Y = cell(4, 1);

% Grid for evaluating 2D KDE mapping
[X_grid, Y_grid] = meshgrid(linspace(0, 1, 50), linspace(0, 1, 50));
pts_grid = [X_grid(:), Y_grid(:)];

for i = 1:4
    cfg = plot_configs(i);
    fpath = fullfile(data_dir, cfg.name);
    
    samples = [];
    log_w = [];
    
    if exist(fpath, 'file')
        tmp = load(fpath);
        if isfield(tmp, 'samples'), samples = tmp.samples; else, vars = fieldnames(tmp); samples = tmp.(vars{1}); end
    end
    
    if cfg.weighted && exist(fpath, 'file')
        wpath = fullfile(data_dir, cfg.wname);
        if exist(wpath, 'file')
            tmp_w = load(wpath);
            if isfield(tmp_w, 'weights'), log_w = tmp_w.weights; else, vars_w = fieldnames(tmp_w); log_w = tmp_w.(vars_w{1}); end
        end
    end
    
    if ~isempty(samples)
        theta_1x = samples(:, 1);
        theta_1y = samples(:, 2);
        
        % Downsample for KDE computation speed
        w_norm = [];
        if cfg.weighted && ~isempty(log_w)
            w_norm = normalize_weights(log_w);
        end
        
        N = length(theta_1x);
        sample_pts = min(N, 10000); % 10k max for KDE evaluation speed
        
        if ~isempty(w_norm)
            idx = randsample(N, sample_pts, true, w_norm);
        else
            idx = randperm(N, sample_pts);
        end
        
        kde_data = [theta_1x(idx), theta_1y(idx)];
        
        % Evaluate 2D KDE
        try
            f_dens_vec = ksdensity(kde_data, pts_grid, 'BoundaryCorrection', 'reflection');
            f_dens = reshape(f_dens_vec, size(X_grid));
            
            all_f_dens{i} = f_dens;
            all_X{i} = X_grid;
            all_Y{i} = Y_grid;
            
            local_max = max(f_dens(:));
            if local_max > global_max_density
                global_max_density = local_max;
            end
        catch e
            fprintf('Warning: KDE failed for %s - %s\n', cfg.title, e.message);
        end
    end
end

% Set global color limits
c_lims = [0, global_max_density * 1.05]; % Adding 5% headroom
if c_lims(2) == 0, c_lims(2) = 1; end % Fallback

% =========================================================================
% 4. Plotting Loop
% =========================================================================

for i = 1:4
    cfg = plot_configs(i);
    
    % Calculate Position [left, bottom, width, height]
    pos_x = MARG_L + (cfg.col - 1) * (plot_w + GAP_W);
    pos_y = 1 - MARG_T - cfg.row * plot_h - (cfg.row - 1) * GAP_H;
    
    ax = axes('Position', [pos_x, pos_y, plot_w, plot_h]);
    
    fpath = fullfile(data_dir, cfg.name);
    
    % Load Samples
    samples = [];
    log_w = [];
    
    if exist(fpath, 'file')
        tmp = load(fpath);
        if isfield(tmp, 'samples')
            samples = tmp.samples;
        else
            vars = fieldnames(tmp);
            samples = tmp.(vars{1});
        end
    end
    
    % Load Weights if applicable
    if cfg.weighted && exist(fpath, 'file')
        wpath = fullfile(data_dir, cfg.wname);
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
    
    hold(ax, 'on'); box(ax, 'on');
    grid(ax, 'on');
    set(ax, 'Layer', 'top', 'GridAlpha', 0.15);
    
    if ~isempty(samples)
        % First 2 dimensions constitute Theta_1
        theta_1x = samples(:, 1);
        theta_1y = samples(:, 2);
        
        max_pts = 50000; 
        N = size(samples, 1);
        if N > max_pts
            if cfg.weighted && ~isempty(log_w)
                w_norm = normalize_weights(log_w);
                idx = randsample(N, max_pts, true, w_norm);
            else
                idx = randperm(N, max_pts);
            end
            theta_1x = theta_1x(idx);
            theta_1y = theta_1y(idx);
        end
        
        % Plot 2D KDE Contours (if successfully computed)
        if ~isempty(all_f_dens{i})
            contour(ax, all_X{i}, all_Y{i}, all_f_dens{i}, 8, 'LineWidth', 0.8);
            colormap(ax, parula); % Using 'parula' for a softer, visually gentle gradient instead of harsh jet
            clim(ax, c_lims);
        end
        
        % 2D Scatter Plot over contours
        scatter(ax, theta_1x, theta_1y, 1.8, cfg.color, 'filled', 'MarkerFaceAlpha', 0.2);
        
        % Plot 4 Ground Truth Points
        scatter(ax, theta_true_x, theta_true_y, 100, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', TRUE_COLOR, 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
        
    else
        text(ax, 0.5, 0.5, 'Data Missing', 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 14);
    end
    
    title(ax, cfg.title, 'Interpreter', 'none', 'FontSize', 12, 'FontWeight', 'bold'); 
    xlim(ax, [0, 1]);
    ylim(ax, [0, 1]);
    set(ax, 'LineWidth', 1.0, 'FontSize', 10);
end

% Add Global Colorbar
% Positioned on the right side of the figure
cb = colorbar(ax, 'Position', [1 - MARG_R + 0.01, MARG_B, 0.015, 1 - MARG_T - MARG_B]);
cb.Label.String = 'Probability Density';
cb.Label.FontSize = 12;
cb.Label.FontWeight = 'bold';
clim(ax, c_lims); % Ensure the last axes applies the right scale for the colorbar

% Manual Global Title
title_txt = '2D Scatter Marginal Distribution: \theta_1';
annotation('textbox', [0, 0.94, 1, 0.05], 'String', title_txt, ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
           'FontSize', 16, 'FontWeight', 'bold');

% Save Figure
save_name = 'P_plot_MU_NU_results.pdf';
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', RESOLUTION_DPI);

% Show Figure
figure(f);
fprintf('\nFinal scatter plot generated and shown.\n');


% =========================================================================
% Local Functions
% =========================================================================

function w_norm = normalize_weights(log_w)
    if isempty(log_w), w_norm=[]; return; end
    log_w = double(log_w(:)); 
    w = exp(log_w - max(log_w)); 
    w_norm = w / sum(w);
end
