clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================

% ---- Read ABBR from simulate_MU.m ----
script_name = 'simulate_MU.m';
if ~exist(script_name, 'file')
    error('File %s not found.', script_name);
end

fid = fopen(script_name, 'r');
for k = 1:9
    tline = fgetl(fid);
    if ~ischar(tline), break; end
end
fclose(fid);

try
    eval(tline); 
    fprintf('Detected ABBR = %s\n', ABBR);
catch
    error('Failed to parse ABBR.');
end

% ---- Initialize System ----
switch ABBR
    case 'TW'
        para = Three_Well(); 
    case 'HB'
        para = Himmelblau();
    case 'AN'
        para = Annulus();
    case 'MW'
        para = Multiple_Well();
end

% ---- Directories ----
data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% =========================================================================
% 2. Load Data
% =========================================================================

fprintf('Loading data...\n');

% 1. Load Samples for Color Limits (Based on PLOT_CHOICE)
% STRICT REQUIREMENT: PLOT_CHOICE must exist in para
if ~isprop(para, 'PLOT_CHOICE')
    error('Error: Property PLOT_CHOICE is missing in class %s.', class(para));
end
choice_idx = para.PLOT_CHOICE;

path_ref = fullfile(data_dir, sprintf('%s_samples_%d.mat', ABBR, choice_idx));
if ~exist(path_ref, 'file')
    error('Data for PLOT_CHOICE=%d not found at %s.', choice_idx, path_ref);
end

tmp = load(path_ref);
ref_field = sprintf('samples_%d', choice_idx);
if ~isfield(tmp, ref_field)
    error('Field %s not found in %s.', ref_field, path_ref);
end
ref_samples = tmp.(ref_field);

% ---- Determine Color Limits based on Selected Samples ----
U_ref = para.compute_potential(ref_samples);

% Define Limits:
c_min = min(U_ref) - 0.2; 
c_max = prctile(U_ref, 99); 

fprintf('Colorbar Limits determined by PLOT_CHOICE=%d:\n', choice_idx);
fprintf('  Min U (Yellow): %.2f\n', c_min);
fprintf('  Max U (White):  %.2f (99th percentile)\n', c_max);

% 2. Load Flow Data
thetas = [0.00, 0.25, 0.50, 0.75, 1.00];
indices = 0:4;
data_flows = struct();

for i = 1:length(indices)
    idx = indices(i);
    theta = thetas(i);
    
    path_samp = fullfile(data_dir, sprintf('%s_samples_%d.mat', ABBR, idx));
    path_w    = fullfile(data_dir, sprintf('%s_weights_%d.mat', ABBR, idx));
    
    if ~exist(path_samp, 'file') || ~exist(path_w, 'file')
        warning('Missing data for index %d. Skipping.', idx);
        data_flows(i).valid = false;
        continue;
    end
    
    tmp_s = load(path_samp);
    field_s = fieldnames(tmp_s);
    samples = tmp_s.(field_s{1});
    
    tmp_w = load(path_w);
    field_w = fieldnames(tmp_w);
    weights = tmp_w.(field_w{1});
    weights = weights(:);
    
    if theta == 0
        name_str = 'Forward KL (\theta=0)';
    elseif theta == 1
        name_str = 'Reverse KL (\theta=1)';
    else
        name_str = sprintf('Jeffreys (\\theta=%.2f)', theta);
    end
    
    data_flows(i).valid = true;
    data_flows(i).theta = theta;
    data_flows(i).name = name_str;
    data_flows(i).samples = samples;
    data_flows(i).weights = weights;
end

% =========================================================================
% 3. Visualization Setup
% =========================================================================

% Width Reduced to 1400
f = figure('Name', ['Jeffreys Flow Comparison - ' para.NAME], ...
       'Color', 'w', 'Position', [50, 200, 2400, 600]);

t = tiledlayout(1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Pre-compute Potential Grid ---
grid_n = 300;
vals_x = linspace(para.X_LIM_PLOT(1), para.X_LIM_PLOT(2), grid_n);
vals_y = linspace(para.Y_LIM_PLOT(1), para.Y_LIM_PLOT(2), grid_n);
[X, Y] = meshgrid(vals_x, vals_y);

points = [X(:), Y(:)];
U_vec = para.compute_potential(points);
U_grid = reshape(U_vec, size(X));

% Clamp for visualization
U_grid(U_grid > c_max) = c_max;
U_grid(U_grid < c_min) = c_min;

% Plotting Parameters
max_plot_pts = 20000; 
marker_size = 2;       
marker_alpha = 0.4; 
marker_color = [0.85, 0.1, 0.1]; % Deep Red

% =========================================================================
% 4. Plotting Subplots
% =========================================================================

for i = 1:length(indices)
    nexttile; 
    setup_subplot(para, X, Y, U_grid, c_min, c_max);
    
    if ~data_flows(i).valid
        text(mean(para.X_LIM_PLOT), mean(para.Y_LIM_PLOT), ...
             'Data Missing', 'HorizontalAlignment', 'center');
        title(sprintf('\\theta=%.2f', thetas(i)));
        continue;
    end
    
    S = data_flows(i).samples;
    W = data_flows(i).weights;
    N_total = size(S, 1);
    
    w_sum = sum(W);
    w_norm = W / w_sum;
    ESS = 1 / sum(w_norm.^2);
    ESS_ratio = (ESS / N_total) * 100;
    
    bias_val = NaN;
    if exist('compute_bias', 'file')
        try
            bias_val = compute_bias(para, S, 1.0, w_norm);
        catch
        end
    end
    
    [S_plot, ~] = filter_in_bounds(S, W, para);
    N_plot = size(S_plot, 1);
    idx = randperm(N_plot, min(N_plot, max_plot_pts));
    
    scatter(S_plot(idx, 1), S_plot(idx, 2), marker_size, marker_color, 'filled', ...
        'MarkerFaceAlpha', marker_alpha);
    
    if isnan(bias_val)
        t_str = sprintf('\\bf %s\nESS=%.1f%%\nBias=NaN', data_flows(i).name, ESS_ratio);
    else
        t_str = sprintf('\\bf %s\nESS=%.1f%%\nBias=%.2e', ...
            data_flows(i).name, ESS_ratio, bias_val);
    end
    % Increased Title Font Size to 14
    title(t_str, 'Interpreter', 'tex', 'FontSize', 14);
end

% =========================================================================
% 5. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, sprintf('%s_error.pdf', ABBR));
fprintf('Saving figure to %s (600 DPI)...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 600);

% =========================================================================
% Helper Functions
% =========================================================================

function [S_in, W_in] = filter_in_bounds(S, W, para)
    mask = S(:,1) >= para.X_LIM_PLOT(1) & S(:,1) <= para.X_LIM_PLOT(2) & ...
           S(:,2) >= para.Y_LIM_PLOT(1) & S(:,2) <= para.Y_LIM_PLOT(2);
    S_in = S(mask, :);
    if ~isempty(W), W_in = W(mask); else, W_in = []; end
end

function setup_subplot(para, X, Y, U_grid, c_min, c_max)
    hold on; box on;
    
    % --- Custom Colormap: Dark Yellow -> Pure White ---
    n_colors = 256;
    
    % Low U (High Prob) = Dark Yellow / Orange-Gold
    c_low  = [1.0, 0.85, 0.2]; 
    % High U (Low Prob) = Pure White
    c_high = [1.0, 1.0, 1.0];
    
    % Linear Interpolation
    r = linspace(c_low(1), c_high(1), n_colors)';
    g = linspace(c_low(2), c_high(2), n_colors)';
    b = linspace(c_low(3), c_high(3), n_colors)';
    custom_map = [r, g, b];

    % 1. Filled Contours
    contourf(X, Y, U_grid, 60, 'LineStyle', 'none');
    
    % 2. Apply Colormap & Limits
    colormap(gca, custom_map);
    clim([c_min, c_max]); 
    
    % 3. Contour Lines (Subtle Grey)
    levels = linspace(c_min, c_max, 15);
    contour(X, Y, U_grid, levels, 'LineColor', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
    
    % Axis Formatting
    
    xlim(para.X_LIM_PLOT);
    ylim(para.Y_LIM_PLOT);
    axis square;
    % Increased Axis Font Size to 12
    set(gca, 'FontSize', 12, 'LineWidth', 1.0, 'Layer', 'top');
    
    % Put scatter on top
    set(gca, 'Children', flipud(get(gca, 'Children')));
end