clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = GMM_3D();
fprintf('Plotting ALL MU samples for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Determine Number of Replicas
% M intervals => M+1 Replicas (Indices 0 to M)
M = para.PT_M;
num_replicas = M + 1;
lambdas = para.BETA_LIST; % Note: In updated GMM_3D, this holds [0...1]

fprintf('Detected M=%d (Total %d Replicas).\n', M, num_replicas);

% =========================================================================
% 2. Visualization Setup
% =========================================================================
% Dynamic figure size based on M
if M > 12
    fig_h = 1200; fig_w = 1600;
else
    fig_h = 400; fig_w = 1600;
end

f = figure('Name', 'Evolution of Replicas (MU)', ...
           'Color', 'w', 'Position', [50, 50, fig_w, fig_h]);

% Flow layout handles any number of plots gracefully
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

% Define Color Gradient (Start -> End)
% Start (Lambda=0): Greenish [0.2, 0.6, 0.2]
% End   (Lambda=1): Blue     [0.2, 0.4, 0.8]
c_start = [0.2, 0.6, 0.2];
c_end   = [0.2, 0.4, 0.8];

% =========================================================================
% 3. Plotting Loop
% =========================================================================

for k = 0 : M
    % 1. Construct File Path
    file_name = sprintf('%s_samples_MU_%d.mat', para.ABBR, k);
    file_path = fullfile(data_dir, file_name);
    
    nexttile;
    
    if ~exist(file_path, 'file')
        text(0, 0, 'Missing', 'HorizontalAlignment', 'center');
        title(sprintf('MU\\_%d', k));
        axis off;
        continue;
    end
    
    % 2. Load Data
    tmp = load(file_path);
    % Variable name is dynamic: samples_MU_0, samples_MU_1, ...
    var_name = sprintf('samples_MU_%d', k);
    if isfield(tmp, var_name)
        samples = tmp.(var_name);
    else
        warning('Variable %s not found in %s', var_name, file_name);
        continue;
    end
    
    % 3. Determine Color & Title
    lam = lambdas(k+1); % MATLAB indices start at 1
    
    % Linear interpolation of color
    this_color = (1 - lam) * c_start + lam * c_end;
    
    % 4. Plot
    plot_projection(samples, para, this_color);
    
    % Title Logic
    if k == 0
        t_str = sprintf('\\mu_0 (Base)\n\\lambda = %.2f', lam);
    elseif k == M
        t_str = sprintf('\\mu_M (Target)\n\\lambda = %.2f', lam);
    else
        t_str = sprintf('\\mu_{%d}\n\\lambda = %.2f', k, lam);
    end
    
    title(t_str, 'Interpreter', 'tex', 'FontSize', 11);
end

% Global Title
title(t, ['Parallel Tempering Stream: ' para.NAME], 'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% 4. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, sprintf('%s_plot_MU_all.png', para.ABBR));
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

% =========================================================================
% Helper Function
% =========================================================================
function plot_projection(samples, para, color)
    % Downsample for speed if needed
    max_pts = 5000; % Reduced for dense plots
    N = size(samples, 1);
    if N > max_pts
        idx = randperm(N, max_pts);
        samples = samples(idx, :);
    end
    
    hold on; box on;
    scatter(samples(:, 1), samples(:, 2), 3, color, 'filled', ...
            'MarkerFaceAlpha', 0.3);
    
    xlim(para.X_LIM_PLOT);
    ylim(para.Y_LIM_PLOT);
    axis square;
    
    % Remove axis labels for inner plots to save space, or keep small
    set(gca, 'FontSize', 9, 'LineWidth', 0.8, 'XTickLabel', [], 'YTickLabel', []);
end