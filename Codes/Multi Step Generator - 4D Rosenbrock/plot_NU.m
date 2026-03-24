clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = Rosenbrock_4D(); 
fprintf('Generating Results Plot for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Load Beta List
beta_file = fullfile(data_dir, sprintf('%s_BETA_LIST.mat', para.ABBR));
if exist(beta_file, 'file')
    tmp = load(beta_file);
    BETA_LIST = tmp.BETA_LIST; % Dimension: [Total_Steps+1 x 1]
    M = length(BETA_LIST) - 1;
    fprintf('Detected M=%d steps from BETA_LIST.\n', M);
else
    error('BETA_LIST file not found. Please run simulation first.');
end

% =========================================================================
% 2. Visualization Setup
% =========================================================================
% Selected Steps to Plot
target_steps = [0, 4, 8, 11];
% Filter steps that are actually within M
target_steps = target_steps(target_steps <= M);
num_cols = length(target_steps);

fprintf('Plotting selected steps: %s\n', mat2str(target_steps));

% Figure Setup
fig_w = 600 * num_cols; 
fig_h = 1500;
f = figure('Name', 'Jeffreys Flow Results (x_1-x_4)', 'Color', 'w', 'Position', [50, 100, fig_w, fig_h]);

% --- Custom Layout Parameters (Manual Control) ---
% Control horizontal gap here. Vertical layout remains fixed.
gap_horz = -0.12;  % <--- [ADJUSTABLE] Horizontal gap between columns
gap_vert = 0.12;   % Fixed vertical gap between rows
marg_l   = 0.0;   % Left margin
marg_r   = 0.0;   % Right margin
marg_b   = 0.08;   % Bottom margin
marg_t   = 0.15;   % Top margin

% Calculate Plot Dimensions (Independent H/V)
% plot_w: Width of a single subplot
plot_w = (1 - marg_l - marg_r - (num_cols - 1) * gap_horz) / num_cols;
% plot_h: Height of a single subplot
plot_h = (1 - marg_b - marg_t - gap_vert) / 2;

% Grid for Background Contours (x1 vs x4)
grid_size = 100; 
x1_lims = para.X1_LIM_PLOT;
x4_lims = para.X4_LIM_PLOT;

x1_vec = linspace(x1_lims(1), x1_lims(2), grid_size);
x4_vec = linspace(x4_lims(1), x4_lims(2), grid_size);
[X1_grid, X4_grid] = meshgrid(x1_vec, x4_vec); % Dimension: [100 x 100] each

% Color Constants
COLOR_MU = [0.2, 0.4, 0.8];       % Blue (Ref)
COLOR_NU = [0.85, 0.1, 0.1];      % Red (Gen)
COLOR_LOW_E = [1.0, 0.85, 0.2];   % Dark Yellow (High Density)
COLOR_HIGH_E = [1.0, 1.0, 1.0];   % White (Low Density)

% Visualization Limit
MAX_PTS = 20000;

% Storage for bias calc (final step)
s_mu_final = []; s_nu_final = []; w_nu_final = [];

% =========================================================================
% 3. Main Plotting Loop (Selected Steps)
% =========================================================================

for i = 1:num_cols
    k = target_steps(i); % Actual step index
    beta_val = BETA_LIST(k + 1);
    
    fprintf('Processing Step %d (Beta=%.3f)...\n', k, beta_val);
    
    % --- A. Load Data ---
    fname_mu = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, k));
    fname_nu = fullfile(data_dir, sprintf('%s_samples_NU_%d.mat', para.ABBR, k));
    fname_w  = fullfile(data_dir, sprintf('%s_weights_NU_%d.mat', para.ABBR, k));
    
    cur_mu = []; cur_nu = []; cur_log_w = [];
    cess_val = -1; ess_val = -1;
    
    if exist(fname_mu, 'file')
        tmp = load(fname_mu); vars = fieldnames(tmp); cur_mu = tmp.(vars{1});
    end
    if exist(fname_nu, 'file') && exist(fname_w, 'file')
        tmp_s = load(fname_nu); if isfield(tmp_s,'samples'), cur_nu=tmp_s.samples; else, vars=fieldnames(tmp_s); cur_nu=tmp_s.(vars{1}); end
        tmp_w = load(fname_w);  
        if isfield(tmp_w,'weights'), cur_log_w=tmp_w.weights; else, vars=fieldnames(tmp_w); cur_log_w=tmp_w.(vars{1}); end
        cur_log_w = cur_log_w(:);
        
        if isfield(tmp_w, 'cess'), cess_val = tmp_w.cess; end
        if isfield(tmp_w, 'ess'),  ess_val  = tmp_w.ess;  end
    end
    
    % --- B. Determine Metrics Strings ---
    cess_str = 'N/A'; ess_str = 'N/A';
    if ~isempty(cur_log_w)
        if ess_val >= 0, ess_str = sprintf('%.1f%%', ess_val * 100); end
        if k == 0, cess_str = 'N/A';
        elseif cess_val >= 0, cess_str = sprintf('%.1f%%', cess_val * 100);
        end
    end

    % --- C. Subsample Data (Strict Match) ---
    n_mu = size(cur_mu, 1);
    n_nu = size(cur_nu, 1);
    
    if n_mu > 0 && n_nu > 0
        N_plot = min([n_mu, n_nu, MAX_PTS]);
        
        % Subsample MU
        idx_mu = randperm(n_mu, N_plot);
        cur_mu_plot = cur_mu(idx_mu, [1, 4]); % Dimension: [N_plot x 2]
        
        % Subsample NU
        idx_nu = randperm(n_nu, N_plot);
        cur_nu_plot = cur_nu(idx_nu, [1, 4]); % Dimension: [N_plot x 2]
        
        % Corresponding Weights for density
        cur_w_density = normalize_weights(cur_log_w);
        cur_w_density = cur_w_density(idx_nu);
        cur_w_density = cur_w_density / sum(cur_w_density); 
    else
        cur_mu_plot = []; cur_nu_plot = []; cur_w_density = [];
        N_plot = 0;
    end

    % --- D. Background Density (Using NU subset) ---
    Z_density = zeros(size(X1_grid));
    bg_samples = cur_nu_plot; 
    bg_weights = cur_w_density;
    
    if isempty(bg_samples) && ~isempty(cur_mu_plot)
        bg_samples = cur_mu_plot; 
        bg_weights = ones(N_plot,1)/N_plot;
    end
    
    if ~isempty(bg_samples)
        grid_pts = [X1_grid(:), X4_grid(:)];
        is_uniform = (std(cur_log_w) < 1e-6); 
        
        if is_uniform
             pdf_vals = ksdensity(bg_samples, grid_pts);
        else
             pdf_vals = ksdensity(bg_samples, grid_pts, 'Weights', bg_weights);
        end
        Z_density = reshape(pdf_vals, grid_size, grid_size);
    end

    % --- E. Plotting (Manual Positions) ---
    
    % Calculate X position for this column
    pos_x = marg_l + (i - 1) * (plot_w + gap_horz);

    % Row 1: MU (Reference) - TOP ROW
    % Y Position: Bottom margin + Height of Row 2 + Vertical Gap
    pos_y1 = marg_b + plot_h + gap_vert; 
    ax1 = axes('Position', [pos_x, pos_y1, plot_w, plot_h]);
    
    plot_step(ax1, cur_mu_plot, COLOR_MU, X1_grid, X4_grid, -Z_density, ...
              COLOR_LOW_E, COLOR_HIGH_E, x1_lims, x4_lims, ...
              sprintf('\\mu_{%d} (Ref)\n\\beta=%.3f', k, beta_val));
    
    xlabel(ax1, 'x_1', 'FontSize', 12, 'FontWeight', 'bold');
    if i == 1
        ylabel(ax1, 'x_4', 'FontSize', 12, 'FontWeight', 'bold');
    end

    % Row 2: NU (Generated) - BOTTOM ROW
    % Y Position: Bottom margin
    pos_y2 = marg_b;
    ax2 = axes('Position', [pos_x, pos_y2, plot_w, plot_h]);
    
    plot_step(ax2, cur_nu_plot, COLOR_NU, X1_grid, X4_grid, -Z_density, ...
              COLOR_LOW_E, COLOR_HIGH_E, x1_lims, x4_lims, ...
              sprintf('\\nu_{%d} (Gen)\nCESS: %s | ESS: %s', k, cess_str, ess_str));
    
    xlabel(ax2, 'x_1', 'FontSize', 12, 'FontWeight', 'bold');
    if i == 1
        ylabel(ax2, 'x_4', 'FontSize', 12, 'FontWeight', 'bold');
    end

    % Save Final Step Data
    if k == M
        s_mu_final = cur_mu; s_nu_final = cur_nu; 
        if ~isempty(cur_log_w), w_nu_final = normalize_weights(cur_log_w); end
    end
    
    drawnow;
end

% =========================================================================
% 4. Final Wrap-up
% =========================================================================
title_str = 'Jeffreys Flow on 4D Rosenbrock (x_1-x_4)';

if exist('compute_bias', 'file') && ~isempty(s_mu_final)
    try
        bias_mu = compute_bias(para, s_mu_final, 1.0, []);
        bias_nu = compute_bias(para, s_nu_final, 1.0, w_nu_final);
        title_str = sprintf('%s - Bias: \\mu=%.2e, \\nu=%.2e', title_str, bias_mu, bias_nu);
    catch; end
end

% Use sgtitle for the main title since we are using manual axes
sgtitle(title_str, 'FontSize', 16, 'FontWeight', 'bold');

save_name = sprintf('%s_plot_results.pdf', para.ABBR);
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);


% =========================================================================
% Local Functions
% =========================================================================

function w_norm = normalize_weights(log_w)
    if isempty(log_w), w_norm=[]; return; end
    w = exp(log_w - max(log_w)); w_norm = w / sum(w);
end

function plot_step(ax, samples, col, X, Y, Z, c_lo, c_hi, xl, yl, txt)
    axes(ax); hold on; box on;
    c_min = min(Z(:)); c_max = max(Z(:)); if c_max==c_min, c_max=c_min+1; end
    n_colors = 256;
    cmap = [linspace(c_lo(1),c_hi(1),n_colors)', linspace(c_lo(2),c_hi(2),n_colors)', linspace(c_lo(3),c_hi(3),n_colors)'];
    contourf(ax, X, Y, Z, linspace(c_min,c_max,30), 'LineStyle', 'none');
    colormap(ax, cmap); clim(ax, [c_min, c_max]);
    
    if ~isempty(samples)
        scatter(ax, samples(:,1), samples(:,2), 3, col, 'filled', 'MarkerFaceAlpha', 0.5);
    else
        text(ax, 0, 0, 'Missing', 'HorizontalAlignment', 'center');
    end
    title(ax, txt, 'Interpreter', 'tex', 'FontSize', 10);
    set(ax, 'XTick', [], 'YTick', [], 'LineWidth', 1.0, 'Layer', 'top');
    xlim(ax, xl); ylim(ax, yl);
    
    % Force strict square aspect ratio
    axis(ax, 'square');
end