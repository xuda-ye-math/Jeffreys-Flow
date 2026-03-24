clear; clc; close all;
rng(27);

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = Nonlinear_8D();
fprintf('Plotting ALL MU samples for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Determine Number of Replicas
% M intervals => M+1 Replicas (Indices 0 to M)
M = para.PT_M;
num_replicas = M + 1;
lambdas = para.BETA_LIST; 

fprintf('Detected M=%d (Total %d Replicas).\n', M, num_replicas);

% =========================================================================
% 2. Visualization Setup
% =========================================================================
f = figure('Name', 'Joint Distributions (Target) - Nonlinear Rastrigin', ...
           'Color', 'w', 'Position', [100, 100, 1600, 400]);

% M is the index of the last replica (Target)
target_idx = M;
file_name = sprintf('%s_samples_MU_%d.mat', para.ABBR, target_idx);
file_path = fullfile(data_dir, file_name);

if ~exist(file_path, 'file')
    fprintf('Target file %s not found. Checking for earlier files...\n', file_name);
    % Fallback: try to find any file if target missing? 
    % But user specifically said "generated mu samples".
    error('Target file %s not found!', file_path);
end

% Load Data
tmp = load(file_path);
var_name = sprintf('samples_MU_%d', target_idx);
if isfield(tmp, var_name)
    samples = tmp.(var_name);
else
    error('Variable %s not found!', var_name);
end

% Downsample if needed
max_pts = 20000;
if size(samples, 1) > max_pts
    idx = randperm(size(samples, 1), max_pts);
    samples = samples(idx, :);
end

% =========================================================================
% 3. Plotting Loop (4 Subplots)
% =========================================================================
% Pairs to plot: (1,2), (2,3), (3,4), (4,5)
% Pairs to plot: (1,2), (3,4), (5,6), (7,8)
pairs = [1, 2; 
         3, 4; 
         5, 6; 
         7, 8];

% Titles using Math notation
titles = {'x_1 - x_2', 'x_3 - x_4', 'x_5 - x_6', 'x_7 - x_8'};

for i = 1:4
    subplot(1, 4, i);
    ax = gca;
    hold on; box on;
    
    dim_x = pairs(i, 1);
    dim_y = pairs(i, 2);
    
    % Scatter Plot
    % Use a nice blue color
    scatter(ax, samples(:, dim_x), samples(:, dim_y), 1.5, [0.2, 0.4, 0.8], 'filled', ...
            'MarkerFaceAlpha', 0.2);
            
    % Formatting
    xlabel(sprintf('x_{%d}', dim_x));
    ylabel(sprintf('x_{%d}', dim_y));
    title(titles{i}, 'Interpreter', 'tex', 'FontSize', 12);
    
    xlim(para.X_LIM_PLOT);
    ylim(para.X_LIM_PLOT);
    axis square;
end

% Global Title
sgtitle(['Target Distribution Joint Projections (' para.NAME ')'], ...
        'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% 4. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, sprintf('%s_plot_MU_joint.png', para.ABBR));
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);