clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
% Ensure the potential folder is in path
addpath(genpath('./potential'));

para = GMM_2D();
fprintf('Plotting ALL MU samples for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Determine Number of Replicas
% M intervals => M+1 Replicas (Indices 0 to M)
% Use BETA_LIST length directly since PT_M is derived/fixed
lambdas = para.BETA_LIST; 
M = length(lambdas) - 1;
num_replicas = M + 1;

fprintf('Detected M=%d (Total %d Replicas).\n', M, num_replicas);

% =========================================================================
% 2. Visualization Setup
% =========================================================================
% Dynamic figure size
% M+1 columns, 2 rows
total_cols = num_replicas;
fig_w = 300 * total_cols;
fig_h = 600; % 2 rows

f = figure('Name', 'Comparison of reSGLD (Naive vs Correct) - 2D', ...
           'Color', 'w', 'Position', [50, 50, fig_w, fig_h]);

% Tiled Layout: 2 Rows x (M+1) Columns
t = tiledlayout(2, total_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Define Color Gradient (Start -> End)
c_start = [0.2, 0.6, 0.2];
c_end   = [0.2, 0.4, 0.8];

% =========================================================================
% 3. Plotting Loop (Function based for reuse)
% =========================================================================

% --- Row 1: Naive (sigma=0) ---
plot_row(1, 'naive', 'reSGLD (Naive)', para, data_dir, lambdas, c_start, c_end);

% --- Row 2: Correct (Variance Corrected) ---
plot_row(2, 'correct', 'reSGLD (Correct)', para, data_dir, lambdas, c_start, c_end);

% Global Title
title(t, ['Variance Correction Comparison: ' para.NAME], 'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% 4. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, sprintf('%s_plot_MU_comparison.png', para.ABBR));
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);


% =========================================================================
% Helper Function
% =========================================================================
function plot_row(row_idx, mode_str, title_prefix, para, data_dir, lambdas, c_start, c_end)
    M = length(lambdas) - 1;
    
    for k = 0 : M
        % Select Tile: Linear index based on row and col
        % Row 1: 1..M+1, Row 2: M+2..2(M+1)
        nexttile((row_idx-1)*(M+1) + k + 1);
        
        file_name = sprintf('%s_samples_%s_MU_%d.mat', para.ABBR, mode_str, k);
        file_path = fullfile(data_dir, file_name);
        
        if ~exist(file_path, 'file')
            text(0, 0, 'Missing', 'HorizontalAlignment', 'center');
            title(sprintf('%s - %d', mode_str, k));
            axis off;
            continue;
        end
        
        % Load Data
        tmp = load(file_path);
        
        % Check for generic field 'samples' first (new convenient standard), then legacy
        % But wait, simulate_MU saves as 'samples_MU_k' inside structure S
        var_name = sprintf('samples_MU_%d', k);
        if isfield(tmp, var_name)
            samples = tmp.(var_name);
        else
            % Fallback just in case
            fnames = fieldnames(tmp);
            samples = tmp.(fnames{1});
        end
        
        % Determine Lambda & Color
        lam = lambdas(k+1); 
        this_color = (1 - lam) * c_start + lam * c_end;
        
        % Compute Bias
        % Pass [0, 0] to compute_potential_mixed inside compute_bias if needed? 
        % compute_bias was fixed to pass [0,0] internally, so we just call it.
        % calculate bias against target measure at lambda=lam
        bias_val = compute_bias(para, samples, lam, []);
        
        % Plot
        max_pts = 5000;
        N_samp = size(samples, 1);
        if N_samp > max_pts
            idx_perm = randperm(N_samp, max_pts);
            samples_plot = samples(idx_perm, :);
        else
            samples_plot = samples;
        end
        
        hold on; box on;
        scatter(samples_plot(:, 1), samples_plot(:, 2), 5, this_color, 'filled', ...
                'MarkerFaceAlpha', 0.4);
        
        xlim(para.X_LIM_PLOT);
        ylim(para.Y_LIM_PLOT);
        axis square;
        grid on;
        set(gca, 'Layer', 'top', 'GridAlpha', 0.1);
        if k > 0
            yticklabels([]);
        end
        
        % Title
        t_str = sprintf('%s (\\lambda=%.2f)\nBias = %.4f', title_prefix, lam, bias_val);
        title(t_str, 'Interpreter', 'tex', 'FontSize', 9);
    end
end
