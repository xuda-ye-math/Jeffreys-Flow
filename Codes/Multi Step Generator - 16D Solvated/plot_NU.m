clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
addpath('./potential');
    
% Let's try to instantiate Solvated_16D if it exists in path, otherwise use hardcoded ABBR.
try
    para = Solvated_16D();
catch
    % Fallback if class not found in path
    para.ABBR = 'SL';
    para.NAME = 'Solvated 16D';
    para.X_LIM_PLOT = [-4.0, 4.0]; 
end

fprintf('Generating 2x4 Scatter Plots for Step 31 %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% =========================================================================
% 2. Visualization Setup
% =========================================================================

% Target Step: 31 (Beta = 1.0)
target_step = 31;
fprintf('Plotting ONLY Step %d (Beta=1.0)\n', target_step);

% Define Columns (Pairs of Dimensions)
% x1-x2, x1-x3, x7-x8, x15-x16
% Note: MATLAB 1-based indexing matches exactly the user description (e.g. x1 -> index 1)
dim_pairs = [1, 2;
             1, 3;
             7, 8;
             15, 16];

num_cols = size(dim_pairs, 1); % 4
num_rows = 2; % MU (Top), NU (Bottom)

% Figure Setup
% Adjust width dynamically
pts_per_plot = 250; 
fig_w = max(min(pts_per_plot * num_cols, 1800), 400); 
fig_h = 600; % Fixed height for 2 rows
f = figure('Name', 'Scatter Plots MU/NU (Step 31)', 'Color', 'w', 'Position', [50, 50, fig_w, fig_h]);

% Tuning Layout
gap_horz = 0.03;
gap_vert = 0.05;
marg_l = 0.05;
marg_r = 0.02;
marg_b = 0.05;
marg_t = 0.10;

plot_w = (1 - marg_l - marg_r - (num_cols-1)*gap_horz) / num_cols;
plot_h = (1 - marg_b - marg_t - (num_rows-1)*gap_vert) / num_rows;

% Scatter Settings
COLOR_MU = [0.2, 0.4, 0.8];       % Blue
COLOR_NU = [0.85, 0.1, 0.1];      % Red
MAX_PTS = 50000; % Plot up to 50k points if available
MARKER_SIZE = 2;
MARKER_ALPHA = 0.5;

% KDE Settings
USE_KDE = true;
grid_size = 100;
COLOR_LOW_E = [1.0, 0.85, 0.2];   % Dark Yellow (High Density)
COLOR_HIGH_E = [1.0, 1.0, 1.0];   % White (Low Density)

% Periodic Indices (Dimensions to wrap)
periodic_dims = [1, 2];
periodic_lims = [-pi, pi];
euclidean_lims = [-4.0, 4.0]; % Standard viewing window for bath

% =========================================================================
% 3. Load Data ONCE
% =========================================================================

k = target_step;
fname_mu = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, k));
fname_nu = fullfile(data_dir, sprintf('%s_samples_NU_%d.mat', para.ABBR, k));

samples_mu = load_samples(fname_mu);
samples_nu = load_samples(fname_nu);

% --- Remove NaNs/Infs ---
if ~isempty(samples_mu), samples_mu = samples_mu(all(isfinite(samples_mu), 2), :); end
if ~isempty(samples_nu), samples_nu = samples_nu(all(isfinite(samples_nu), 2), :); end

% Determine common count to plot
n_mu = size(samples_mu, 1);
n_nu = size(samples_nu, 1);
n_plot = min([n_mu, n_nu, MAX_PTS]);

if n_plot == 0
    error('Step %d: No valid samples found (MU=%d, NU=%d). Aborting.', k, n_mu, n_nu);
else
    % Subsample BOTH to exactly n_plot
    pts_mu = subsample(samples_mu, n_plot);
    pts_nu = subsample(samples_nu, n_plot);
end

% --- Periodic Wrap Application ---
% Only wrap dimensions 1 and 2 if they exist in the dataset
for d = periodic_dims
    if d <= size(pts_mu, 2)
        pts_mu(:, d) = mod(pts_mu(:, d) + pi, 2*pi) - pi;
    end
    if d <= size(pts_nu, 2)
        pts_nu(:, d) = mod(pts_nu(:, d) + pi, 2*pi) - pi;
    end
end

% =========================================================================
% 4. Plotting Loop (Columns)
% =========================================================================

for col_idx = 1:num_cols
    d1 = dim_pairs(col_idx, 1);
    d2 = dim_pairs(col_idx, 2);
    
    % Determine limits for this pair
    
    % Axis 1 Limits
    if ismember(d1, periodic_dims)
        lims_x = periodic_lims;
    else
        lims_x = euclidean_lims;
    end
    
    % Axis 2 Limits
    if ismember(d2, periodic_dims)
        lims_y = periodic_lims;
    else
        lims_y = euclidean_lims;
    end

    pos_x = marg_l + (col_idx-1)*(plot_w + gap_horz);
    
    % --- KDE Computation ---
    Z_mu = zeros(grid_size, grid_size);
    Z_nu = zeros(grid_size, grid_size);
    
    x_vec = linspace(lims_x(1), lims_x(2), grid_size);
    y_vec = linspace(lims_y(1), lims_y(2), grid_size);
    [X_grid, Y_grid] = meshgrid(x_vec, y_vec);
    grid_pts = [X_grid(:), Y_grid(:)];
    
    if USE_KDE
        % Compute KDE for MU
        try
            if ~isempty(pts_mu)
                pdf_vals = ksdensity(stats_pts(pts_mu(:, [d1, d2])), grid_pts, 'Bandwidth', []); % Use valid points
                Z_mu = reshape(pdf_vals, grid_size, grid_size);
            end
        catch
            warning('KDE failed for MU %d-%d', d1, d2);
        end
        
        % Compute KDE for NU
        try
             if ~isempty(pts_nu)
                pdf_vals = ksdensity(stats_pts(pts_nu(:, [d1, d2])), grid_pts, 'Bandwidth', []);
                Z_nu = reshape(pdf_vals, grid_size, grid_size);
             end
        catch
            warning('KDE failed for NU %d-%d', d1, d2);
        end
    end

    % --- Row 1: MU (Target) ---
    pos_y_row1 = marg_b + plot_h + gap_vert;
    ax_mu = axes('Position', [pos_x, pos_y_row1, plot_w, plot_h]);
    
    plot_step(ax_mu, pts_mu, d1, d2, COLOR_MU, X_grid, Y_grid, -Z_mu, ...
              COLOR_LOW_E, COLOR_HIGH_E, lims_x, lims_y, MARKER_SIZE, MARKER_ALPHA);
    
    % Title (Ref)
    title(ax_mu, sprintf('\\mu (Ref) x_{%d}-x_{%d}', d1, d2), 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'tex');
    
    ylabel(ax_mu, sprintf('x_{%d}', d2), 'FontSize', 12);
    if col_idx ~= 1
        set(ax_mu, 'YTickLabel', []);
    end
    set(ax_mu, 'XTickLabel', []);
    
    % --- Row 2: NU (Generated) ---
    pos_y_row2 = marg_b;
    ax_nu = axes('Position', [pos_x, pos_y_row2, plot_w, plot_h]);
    
    plot_step(ax_nu, pts_nu, d1, d2, COLOR_NU, X_grid, Y_grid, -Z_nu, ...
              COLOR_LOW_E, COLOR_HIGH_E, lims_x, lims_y, MARKER_SIZE, MARKER_ALPHA);
    
    % Title (Gen)
    title(ax_nu, sprintf('\\nu (Gen) x_{%d}-x_{%d}', d1, d2), 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'tex');
    
    ylabel(ax_nu, sprintf('x_{%d}', d2), 'FontSize', 12);
    if col_idx ~= 1
        set(ax_nu, 'YTickLabel', []);
    end
    
    xlabel(ax_nu, sprintf('x_{%d}', d1)); 
end

% =========================================================================
% 5. Save
% =========================================================================
sgtitle(sprintf('Joint Distributions - 16D %s', para.NAME), 'FontSize', 16, 'FontWeight', 'bold');

save_name = sprintf('%s_plot_results.png', para.ABBR);
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

% =========================================================================
% Helpers
% =========================================================================

function samples = load_samples(fname)
    samples = [];
    if exist(fname, 'file')
        tmp = load(fname);
        if isfield(tmp, 'samples')
            samples = tmp.samples;
        else
            % Fallback for other variable names
            fnames = fieldnames(tmp);
            samples = tmp.(fnames{1});
        end
    else
        warning('File not found: %s', fname);
    end
end

function pts = subsample(samples, max_n)
    if isempty(samples)
        pts = [];
        return;
    end
    N = size(samples, 1);
    if N > max_n
        idx = randperm(N, max_n);
        pts = samples(idx, :);
    else
        pts = samples;
    end
end

function pts_out = stats_pts(pts_in)
    % Remove strict duplicates for ksdensity stability if needed
    % (ksdensity usually handles them, but just in case)
    pts_out = pts_in;
end

function plot_step(ax, pts, d1, d2, col, X, Y, Z, c_lo, c_hi, lims_x, lims_y, m_size, m_alpha)
    axes(ax); hold on; box on;
    
    % Contour Background
    if any(Z(:) ~= 0)
        c_min = min(Z(:)); c_max = max(Z(:)); 
        if c_max==c_min, c_max=c_min+1; end
        
        n_colors = 256;
        cmap = [linspace(c_lo(1),c_hi(1),n_colors)', linspace(c_lo(2),c_hi(2),n_colors)', linspace(c_lo(3),c_hi(3),n_colors)'];
        
        contourf(ax, X, Y, Z, linspace(c_min,c_max,30), 'LineStyle', 'none');
        colormap(ax, cmap); clim(ax, [c_min, c_max]);
    end
    
    % Scatter Points
    if ~isempty(pts)
        scatter(pts(:, d1), pts(:, d2), m_size, col, 'filled', 'MarkerFaceAlpha', m_alpha);
    else
        text(0, 0, 'No Data', 'HorizontalAlignment', 'center');
    end
    
    xlim(lims_x); ylim(lims_y);
    axis square;
    grid on;
    set(ax, 'Layer', 'top'); % Grid on top
end