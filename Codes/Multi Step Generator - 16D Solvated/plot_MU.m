clear; clc; close all;
rng(27);

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = Solvated_16D();   % CHANGED TO SOLVATED
fprintf('Plotting ALL MU samples for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Determine Number of Replicas from Saved BETA_LIST
% This is better than restarting Solvated_16D because simulate_MU might have used hybrid ladder
beta_file = fullfile(data_dir, sprintf('%s_BETA_LIST.mat', para.ABBR));
if exist(beta_file, 'file')
    tmp = load(beta_file);
    lambdas = tmp.BETA_LIST;
    M = length(lambdas) - 1;
    fprintf('Loaded BETA_LIST from file. M=%d\n', M);
else
    fprintf('Warning: BETA_LIST file not found. Using class default.\n');
    M = para.PT_M;
    lambdas = para.BETA_LIST; 
end

num_replicas = M + 1;

fprintf('Detected M=%d (Total %d Replicas).\n', M, num_replicas);

% =========================================================================
% 2. Visualization Setup
% =========================================================================
% Calculate figure size based on replicas
% Width: 200px per replica
% Height: 200px per row (4 rows)
fig_w = max(1200, 200 * num_replicas);
fig_h = 200 * 4;

f = figure('Name', 'Joint Distributions Evolution - Solvated 16D', ...
           'Color', 'w', 'Position', [50, 50, fig_w, fig_h]);

% Pairs to plot: (1,2) [Grid], (3,4) [Bath], (5,6) [Bath], (7,8) [Bath]
pairs = [1, 2; 3, 4; 5, 6; 7, 8];
row_titles = {'Dims 1-2 (Grid)', 'Dims 3-4 (Bath)', 'Dims 5-6 (Bath)', 'Dims 7-8 (Bath)'};
num_rows = 4;

% =========================================================================
% 3. Plotting Loop (4 Rows x N Columns)
% =========================================================================

for k = 0 : M
    % 1. Load Data for Replica k
    file_name = sprintf('%s_samples_MU_%d.mat', para.ABBR, k);
    file_path = fullfile(data_dir, file_name);
    
    if ~exist(file_path, 'file')
        continue;
    end
    
    tmp = load(file_path);
    var_name = sprintf('samples_MU_%d', k);
    if isfield(tmp, var_name)
        samples = tmp.(var_name);
    else
        continue;
    end
    
    % Downsample
    max_pts = 2000; % Reduced points per plot for large grid
    if size(samples, 1) > max_pts
        idx = randperm(size(samples, 1), max_pts);
        samples = samples(idx, :);
    end
    
    % 2. Plot Columns (Replicas)
    col_idx = k + 1;
    beta_val = lambdas(col_idx);
    
    for r = 1 : num_rows
        dim_x = pairs(r, 1);
        dim_y = pairs(r, 2);
        
        % Subplot Index: (row-1)*cols + col
        sp_idx = (r - 1) * num_replicas + col_idx;
        subplot(num_rows, num_replicas, sp_idx);
        
        ax = gca;
        hold on; box on;
        
        % Scatter
        scatter(ax, samples(:, dim_x), samples(:, dim_y), 2.0, [0.2, 0.4, 0.8], 'filled', ...
                'MarkerFaceAlpha', 0.4);
        
        % Limits
        if r == 1
            xlim(para.X_PERI_LIM);
            ylim(para.X_PERI_LIM);
        else
            xlim(para.X_FREE_LIM_PLOT);
            ylim(para.X_FREE_LIM_PLOT);
        end
        
        % Axis Labels & Ticks
        axis square;
        set(gca, 'XTickLabel', [], 'YTickLabel', []); % Clean look
        
        % Column Title (Top Row Only)
        if r == 1
            title(sprintf('\\lambda = %.2f', beta_val), 'FontSize', 10);
        end
        
        % Row Label (Left Column Only)
        if col_idx == 1
            ylabel(row_titles{r}, 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
    
    fprintf('  Processed Replica %d / %d (Lambda=%.2f)\n', k, M, beta_val);
end

% Global Title
sgtitle(['Evolution of Solvated 16D Distribution (' para.NAME ')'], ...
        'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% 4. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, sprintf('%s_plot_MU_evolution.png', para.ABBR));
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);