clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================

% Initialize Problem
para = Periodic_Well();
ABBR = para.ABBR;

% Directories
data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('Processing Convergence Analysis for %s (%s)...\n', para.NAME, ABBR);

% Exponents for Sample Sizes: N = 4^k
exponents = [8, 9, 10, 11, 12];
sample_sizes = 4.^exponents;
num_pts = length(exponents);

% Pre-allocate arrays
list_N    = zeros(1, num_pts);
list_ESS  = zeros(1, num_pts);
list_Bias = zeros(1, num_pts);

% Samples for Visualization (We will store the largest set for the Left Plot)
plot_samples = [];
plot_weights = [];

% =========================================================================
% 2. Data Processing Loop
% =========================================================================

for i = 1:num_pts
    k = exponents(i);
    N = sample_sizes(i);
    list_N(i) = N;
    
    fprintf('  > Processing N = 4^%d (%d)... ', k, N);
    
    % Construct File Paths
    file_samp = fullfile(data_dir, sprintf('%s_samples_%d.mat', ABBR, k));
    file_w    = fullfile(data_dir, sprintf('%s_weights_%d.mat', ABBR, k));
    
    if ~exist(file_samp, 'file') || ~exist(file_w, 'file')
        warning('Missing data for exponent %d. Skipping.', k);
        list_ESS(i) = NaN;
        list_Bias(i) = NaN;
        continue;
    end
    
    % Load Samples
    tmp_s = load(file_samp);
    key_s = sprintf('samples_%d', k);
    samples = tmp_s.(key_s);
    
    % Load Weights
    tmp_w = load(file_w);
    key_w = sprintf('weights_%d', k);
    weights = tmp_w.(key_w);
    weights = weights(:); % Column vector
    
    % 1. Calculate ESS
    w_sum = sum(weights);
    w_norm = weights / w_sum;
    ESS_val = 1 / sum(w_norm.^2);
    ESS_perc = (ESS_val / N) * 100;
    list_ESS(i) = ESS_perc;
    
    % 2. Calculate Bias
    if exist('compute_bias', 'file')
        bias_val = compute_bias(para, samples, 1.0, w_norm);
    else
        bias_val = NaN;
    end
    list_Bias(i) = bias_val;
    
    fprintf('ESS: %.2f%% | Bias: %.2e\n', ESS_perc, bias_val);
    
    % Store the largest dataset for plotting (k=12)
    if k == 12
        plot_samples = samples;
        plot_weights = weights;
    end
end

% Fallback if k=12 missing
if isempty(plot_samples) && exist('samples', 'var')
    plot_samples = samples;
    plot_weights = weights;
end

% =========================================================================
% 3. Visualization
% =========================================================================

f = figure('Name', ['Convergence Analysis - ' para.NAME], ...
           'Color', 'w', 'Position', [100, 200, 1200, 500]);

% Ensure renderer is set to handle both transparency and lines correctly
set(f, 'Renderer', 'painters'); 

t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'normal');

% --- Subplot 1: Potential Landscape & Samples ---
nexttile;
setup_potential_plot(para, plot_samples, plot_weights);
% Fixed Title: Single Line
title(sprintf('\\bf Distribution Visualization (N = 4^{12})'), 'FontSize', 14, 'Interpreter', 'tex');

% --- Subplot 2: Convergence Metrics (Dual Axis) ---
nexttile;
setup_convergence_plot(list_N, list_Bias, list_ESS);
title(sprintf('\\bf Convergence Metrics'), 'FontSize', 14, 'Interpreter', 'tex');

% =========================================================================
% 4. Save Figure
% =========================================================================
% Force a draw update before saving to ensure lines are rendered
drawnow;

save_path = fullfile(fig_dir, sprintf('%s_convergence.pdf', ABBR));
fprintf('Saving figure to %s (600 DPI)...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 600);

% =========================================================================
% Helper Functions
% =========================================================================

function setup_potential_plot(para, S, W)
    hold on; box on;
    
    % 1. Compute Potential Grid
    grid_n = 300;
    vals_x = linspace(para.X_LIM_PLOT(1), para.X_LIM_PLOT(2), grid_n);
    vals_y = linspace(para.Y_LIM_PLOT(1), para.Y_LIM_PLOT(2), grid_n);
    [X, Y] = meshgrid(vals_x, vals_y);
    points = [X(:), Y(:)];
    U_vec = para.compute_potential(points);
    U_grid = reshape(U_vec, size(X));
    
    % 2. Limits & Colormap
    if ~isempty(S)
        U_samp = para.compute_potential(S);
        c_min = min(U_samp) - 0.2;
        c_max = prctile(U_samp, 98);
    else
        c_min = min(U_vec);
        c_max = max(U_vec);
    end
    
    U_grid(U_grid > c_max) = c_max;
    U_grid(U_grid < c_min) = c_min;
    
    n_colors = 256;
    c_low  = [1.0, 0.85, 0.2]; 
    c_high = [1.0, 1.0, 1.0];
    r = linspace(c_low(1), c_high(1), n_colors)';
    g = linspace(c_low(2), c_high(2), n_colors)';
    b = linspace(c_low(3), c_high(3), n_colors)';
    custom_map = [r, g, b];
    
    contourf(X, Y, U_grid, 60, 'LineStyle', 'none');
    colormap(gca, custom_map);
    clim([c_min, c_max]);
    
    levels = linspace(c_min, c_max, 15);
    contour(X, Y, U_grid, levels, 'LineColor', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
    
    % 3. Scatter Plot
    if ~isempty(S)
        max_plot_pts = 20000;
        N_total = size(S, 1);
        idx = randperm(N_total, min(N_total, max_plot_pts));
        
        scatter(S(idx, 1), S(idx, 2), 3, [0.85, 0.1, 0.1], 'filled', ...
            'MarkerFaceAlpha', 0.5);
    end
    
    xlabel('x'); ylabel('y');
    xlim(para.X_LIM_PLOT);
    ylim(para.Y_LIM_PLOT);
    axis square;
    set(gca, 'FontSize', 12, 'LineWidth', 1.0, 'Layer', 'top');
end

function setup_convergence_plot(N, Bias, ESS)
    % Remove NaNs
    mask = ~isnan(Bias) & ~isnan(ESS);
    N = N(mask);
    Bias = Bias(mask);
    ESS = ESS(mask);
    
    % --- Left Axis: L2 Bias (Base 2 Log Scale) ---
    yyaxis left
    
    % Plot using standard loglog first
    p1 = loglog(N, Bias, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
        'Color', [0.2, 0.4, 0.8], 'MarkerFaceColor', [0.2, 0.4, 0.8]);
    
    ylabel('L2 Bias (Log Scale, Base 2)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    set(gca, 'YColor', 'k'); % Black axis text
    
    % Custom Y-Ticks for Base 2
    if ~isempty(Bias)
        min_b = min(Bias);
        max_b = max(Bias);
        % Find nearest powers of 2 enclosing the data
        min_pow = floor(log2(min_b));
        max_pow = ceil(log2(max_b));
        
        % Generate ticks: 2^k
        ticks_pow = min_pow : 1 : max_pow; 
        ticks_val = 2.^ticks_pow;
        
        % Filter ticks to avoid overcrowding if range is too large
        if length(ticks_val) > 10
            ticks_pow = min_pow : 2 : max_pow;
            ticks_val = 2.^ticks_pow;
        end
        
        ylim([2^(min_pow-0.5), 2^(max_pow+0.5)]); % Add some padding
        yticks(ticks_val);
        
        % Create Labels: "2^{-10}"
        labels = arrayfun(@(p) sprintf('2^{%d}', p), ticks_pow, 'UniformOutput', false);
        yticklabels(labels);
    end
    grid on;
    
    % --- Right Axis: ESS ---
    yyaxis right
    p2 = semilogx(N, ESS, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
        'Color', [0.85, 0.3, 0.1], 'MarkerFaceColor', [0.85, 0.3, 0.1]);
    
    ylabel('Effective Sample Size (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    ylim([0, 105]);
    set(gca, 'YColor', 'k'); % Black axis text
    
    % --- X-Axis Formatting ---
    xlabel('Sample Size (N)', 'FontSize', 12, 'FontWeight', 'bold');
    xlim([min(N)*0.8, max(N)*1.2]);
    
    % Custom Ticks for powers of 4
    xticks(N);
    xticklabels(arrayfun(@(n) sprintf('4^{%d}', round(log(n)/log(4))), N, 'UniformOutput', false));
    
    set(gca, 'FontSize', 12, 'LineWidth', 1.0);
    
    % --- Legend ---
    % Explicitly reference plot objects to ensure legend appears
    lgd = legend([p1, p2], {'L2 Bias', 'ESS %'}, 'Location', 'southwest', 'FontSize', 13);
    lgd.ItemTokenSize = [25, 18];
end