clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = GMM_3D(); 
fprintf('Generating Results Plot for %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Load Beta List
beta_file = fullfile(data_dir, sprintf('%s_BETA_LIST.mat', para.ABBR));
if exist(beta_file, 'file')
    tmp = load(beta_file);
    BETA_LIST = tmp.BETA_LIST;
    M = length(BETA_LIST) - 1;
    fprintf('Detected M=%d steps from BETA_LIST.\n', M);
else
    error('BETA_LIST file not found. Please run simulation first.');
end

% =========================================================================
% 2. Visualization Setup (Isometric Projection)
% =========================================================================
% Projection Basis for Plane x+y+z=0
n = [1; 1; 1]; n = n / norm(n);
u = [1; -1; 0]; u = u / norm(u);
v = cross(n, u);

fprintf('Projection Plane: x+y+z=0\n');

% Figure Setup
fig_w = 280 * (M + 1); 
fig_h = 600;
f = figure('Name', 'Jeffreys Flow Results', 'Color', 'w', 'Position', [50, 100, fig_w, fig_h]);

t = tiledlayout(2, M + 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Grid for Background Contours (Defined on the 2D u-v plane)
grid_size = 100; 
uv_lims = para.X_LIM_PLOT; 
u_vec = linspace(uv_lims(1), uv_lims(2), grid_size);
v_vec = linspace(uv_lims(1), uv_lims(2), grid_size);
[U_grid, V_grid] = meshgrid(u_vec, v_vec);

% Color Constants
COLOR_MU = [0.2, 0.4, 0.8];       % Blue (Ref)
COLOR_NU = [0.85, 0.1, 0.1];      % Red (Gen)
COLOR_LOW_E = [1.0, 0.85, 0.2];   % Dark Yellow (High Density)
COLOR_HIGH_E = [1.0, 1.0, 1.0];   % White (Low Density)

% Storage for final step data (3D) for bias calc
s_mu_final = []; s_nu_final = []; w_nu_final = [];

% =========================================================================
% 3. Main Plotting Loop (Iterative Loading & Clearing)
% =========================================================================
for k = 0 : M
    beta_val = BETA_LIST(k + 1);
    fprintf('Processing Step %d (Beta=%.3f)...\n', k, beta_val);
    
    % --- A. Load Data for Current Step ONLY ---
    fname_mu = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, k));
    fname_nu = fullfile(data_dir, sprintf('%s_samples_NU_%d.mat', para.ABBR, k));
    fname_w  = fullfile(data_dir, sprintf('%s_weights_NU_%d.mat', para.ABBR, k));
    
    cur_mu_3d = []; cur_nu_3d = []; cur_w = []; ess_str = 'N/A';
    
    if exist(fname_mu, 'file')
        tmp = load(fname_mu); vars = fieldnames(tmp); cur_mu_3d = tmp.(vars{1});
    end
    if exist(fname_nu, 'file') && exist(fname_w, 'file')
        tmp_s = load(fname_nu); if isfield(tmp_s,'samples'), cur_nu_3d=tmp_s.samples; else, vars=fieldnames(tmp_s); cur_nu_3d=tmp_s.(vars{1}); end
        tmp_w = load(fname_w);  if isfield(tmp_w,'weights'), log_w=tmp_w.weights; else, vars=fieldnames(tmp_w); log_w=tmp_w.(vars{1}); end
        
        w_norm = exp(log_w - max(log_w)); cur_w = w_norm / sum(w_norm);
        ESS = 1 / sum(cur_w.^2); ESS_perc = (ESS / length(cur_w)) * 100;
        ess_str = sprintf('%.1f%%', ESS_perc);
    end

    % --- B. Project Samples to 2D Plane (u, v) ---
    cur_mu_proj = []; if ~isempty(cur_mu_3d), cur_mu_proj = cur_mu_3d * [u, v]; end
    cur_nu_proj = []; if ~isempty(cur_nu_3d), cur_nu_proj = cur_nu_3d * [u, v]; end

    % --- C. Compute KDE Density Background (Using FULL NU samples) ---
    Z_density = zeros(size(U_grid));
    
    bg_samples = [];
    bg_weights = [];
    
    if ~isempty(cur_nu_proj)
        bg_samples = cur_nu_proj;
        bg_weights = cur_w; 
    elseif ~isempty(cur_mu_proj)
        bg_samples = cur_mu_proj;
        bg_weights = ones(size(cur_mu_proj,1),1) / size(cur_mu_proj,1);
    end
    
    if ~isempty(bg_samples)
        % Compute 2D KDE using FULL data (No Downsampling)
        % ksdensity evaluates PDF at grid points [U_grid, V_grid]
        grid_pts = [U_grid(:), V_grid(:)];
        
        if ~isempty(bg_weights)
             % Use weights for correct density estimation of importance samples
             w_in = bg_weights / sum(bg_weights);
             pdf_vals = ksdensity(bg_samples, grid_pts, 'Weights', w_in);
        else
             pdf_vals = ksdensity(bg_samples, grid_pts);
        end
        
        Z_density = reshape(pdf_vals, grid_size, grid_size);
    end

    % --- D. Plot Row 1 (MU - Projected) ---
    ax1 = nexttile(k + 1);
    % Plot -Density so high density is low value (Yellow)
    plot_step(ax1, cur_mu_proj, COLOR_MU, U_grid, V_grid, -Z_density, ...
              COLOR_LOW_E, COLOR_HIGH_E, uv_lims, sprintf('\\mu_{%d} (Ref)\n\\beta=%.3f', k, beta_val));
    if k == 0, ylabel(ax1, 'Ref (Iso-Proj)', 'FontWeight', 'bold', 'FontSize', 12); end

    % --- E. Plot Row 2 (NU - Projected) ---
    ax2 = nexttile(M + 1 + k + 1);
    plot_step(ax2, cur_nu_proj, COLOR_NU, U_grid, V_grid, -Z_density, ...
              COLOR_LOW_E, COLOR_HIGH_E, uv_lims, sprintf('\\nu_{%d} (Gen)\nESS: %s', k, ess_str));
    if k == 0, ylabel(ax2, 'Gen (Iso-Proj)', 'FontWeight', 'bold', 'FontSize', 12); end

    % --- F. Save Final Step Data (3D) & Clear Memory ---
    if k == M
        s_mu_final = cur_mu_3d; s_nu_final = cur_nu_3d; w_nu_final = cur_w;
    end
    
    clear cur_mu_3d cur_nu_3d cur_w log_w w_norm cur_mu_proj cur_nu_proj Z_density bg_samples bg_weights tmp tmp_s tmp_w; 
    drawnow; 
end

% =========================================================================
% 4. Compute Bias (Final Step - using 3D data) & Save
% =========================================================================
bias_mu = 0; bias_nu = 0;
if ~isempty(s_mu_final) && ~isempty(s_nu_final)
    fprintf('Computing Analytical Bias for Final Step (Beta=1) using 3D data...\n');
    bias_mu = compute_bias(para, s_mu_final, 1.0, []);
    bias_nu = compute_bias(para, s_nu_final, 1.0, w_nu_final);
    fprintf('  Mu vs Truth: %.4e\n', bias_mu);
    fprintf('  Nu vs Truth: %.4e\n', bias_nu);
end

title_str = sprintf('Jeffreys Flow on 3D GMM - Bias of \\mu / \\nu = %.4e / %.4e', bias_mu, bias_nu);
title(t, title_str, 'FontSize', 16, 'FontWeight', 'bold');

save_name = sprintf('%s_plot_results.pdf', para.ABBR);
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

% =========================================================================
% Local Helper Functions
% =========================================================================

function plot_step(ax, samples, scatter_color, X, Y, Z, c_low, c_high, plot_lims, title_txt)
    axes(ax); hold on; box on;
    
    % 1. Background Contours (Smoothed Density)
    c_min = min(Z(:));
    c_max = max(Z(:));
    if c_max == c_min, c_max = c_min + 1; end
    
    n_colors = 256;
    custom_map = [linspace(c_low(1), c_high(1), n_colors)', ...
                  linspace(c_low(2), c_high(2), n_colors)', ...
                  linspace(c_low(3), c_high(3), n_colors)'];
    
    levels = linspace(c_min, c_max, 30);
    contourf(ax, X, Y, Z, levels, 'LineStyle', 'none');
    colormap(ax, custom_map);
    clim(ax, [c_min, c_max]); 
    
    % 2. Scatter Plot (Subset)
    if ~isempty(samples)
        max_pts = 4000; % Keep scatter subset
        N = size(samples, 1);
        if N > max_pts
            % Uniform random subset for plotting is sufficient for visualization
            idx = randperm(N, max_pts); 
            samples = samples(idx, :);
        end
        scatter(ax, samples(:, 1), samples(:, 2), 5, scatter_color, 'filled', 'MarkerFaceAlpha', 0.5);
    else
        text(ax, 0, 0, 'Missing', 'HorizontalAlignment', 'center', 'Color', 'k');
    end
    
    % Formatting
    title(ax, title_txt, 'Interpreter', 'tex', 'FontSize', 12);
    set(ax, 'XTick', [], 'YTick', [], 'LineWidth', 1.0, 'Layer', 'top');
    xlim(ax, plot_lims); ylim(ax, plot_lims); axis(ax, 'square');
end