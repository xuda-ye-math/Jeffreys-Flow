clear; clc; close all;
addpath('./potential');

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
para = Nonlinear_8D(); 
fprintf('Generating Comparison Plots: Parallel Tempering vs Jeffreys Flow (%s)...\n', para.ABBR);

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
    error('BETA_LIST file not found.');
end

% =========================================================================
% 2. Visualization Setup
% =========================================================================
k = M; % Target Step
fprintf('Visualizing Final Step %d...\n', k);

% Colors
COLOR_PT   = [0.2, 0.2, 0.2];     % Black/Grey (Baseline)
COLOR_FLOW = [0.85, 0.1, 0.1];    % Red (Ours)

% =========================================================================
% 3. Load Data
% =========================================================================
fname_mu = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, k));
fname_nu = fullfile(data_dir, sprintf('%s_samples_NU_%d.mat', para.ABBR, k));
fname_w  = fullfile(data_dir, sprintf('%s_weights_NU_%d.mat', para.ABBR, k));

samples_pt = []; samples_flow = []; log_w = [];

% Load PT (MU)
if exist(fname_mu, 'file')
    tmp = load(fname_mu); vars = fieldnames(tmp); samples_pt = tmp.(vars{1});
    fprintf('Loaded PT samples (MU): %d\n', size(samples_pt,1));
else
    warning('Missing PT samples (MU). Comparison will be incomplete.');
end

% Load Flow (NU)
if exist(fname_nu, 'file')
    tmp = load(fname_nu); 
    if isfield(tmp,'samples'), samples_flow=tmp.samples; else, vars=fieldnames(tmp); samples_flow=tmp.(vars{1}); end
    fprintf('Loaded Flow samples (NU): %d\n', size(samples_flow,1));
else
    error('Missing Flow samples (NU). Cannot plot.');
end

% Load Weights
if exist(fname_w, 'file')
    tmp = load(fname_w);
    if isfield(tmp,'weights'), log_w=tmp.weights; else, vars=fieldnames(tmp); log_w=tmp.(vars{1}); end
    log_w = log_w(:);
end

% Normalize Weights
w_flow = [];
if ~isempty(log_w)
    w_flow = normalize_weights(log_w);
else
    w_flow = ones(size(samples_flow,1), 1) / size(samples_flow,1);
end

% =========================================================================
% 4. Compute Energy (Potential U(x))
% =========================================================================
fprintf('Computing Potentials...\n');
U_pt = []; U_flow = [];

if ~isempty(samples_pt)
    U_pt = para.compute_potential(samples_pt);
end

if ~isempty(samples_flow)
    U_flow = para.compute_potential(samples_flow);
end

% --- Layout Refactoring (Dynamic Figure Size for Square Plots) ---
% Tight Margins
gap_horz = 0.08;  
marg_l   = 0.08;   
marg_r   = 0.02;   
marg_b   = 0.12;   
marg_t   = 0.12;   

num_cols = 2;

% Calculate Target Figure Dimensions to ensure Square Subplots without Gaps
% Let H_fig be fixed.
fig_h = 500;

% We want the plotting area height (fraction):
h_frac = 1 - marg_b - marg_t;

% We want the plotting area width (fraction) per subplot to equal h_frac * (fig_h / fig_w)
% But simpler: Let W_plot_px = H_plot_px
% H_plot_px = fig_h * h_frac
% Total W_needed_px = num_cols * H_plot_px / (1 - gap_fraction_loss) ... slightly complex
% Alternatively:
% W_plot_frac = (1 - marg_l - marg_r - (num_cols-1)*gap_horz) / num_cols
% We need: fig_w * W_plot_frac = fig_h * h_frac
% => fig_w = fig_h * h_frac / W_plot_frac

W_plot_frac_ideal = (1 - marg_l - marg_r - (num_cols - 1) * gap_horz) / num_cols;
fig_w = fig_h * h_frac / W_plot_frac_ideal;

fprintf('Dynamic Figure Resize: [W, H] = [%.1f, %.1f]\n', fig_w, fig_h);

% Re-create figure with optimal size
f = figure('Name', 'PT vs Flow Verification', 'Color', 'w', 'Position', [100, 100, fig_w, fig_h]);

% Recalculate positions based on new Figure W/H
% (Now plot_w and plot_h in normalized units should correspond to square pixels)
plot_w = (1 - marg_l - marg_r - (num_cols - 1) * gap_horz) / num_cols;
plot_h = 1 - marg_b - marg_t;

% Define Positions
pos_x1 = marg_l;
pos_x2 = marg_l + plot_w + gap_horz;

pos1 = [pos_x1, marg_b, plot_w, plot_h]; 
pos2 = [pos_x2, marg_b, plot_w, plot_h];  

% =========================================================================
% 5. Plot 1: Energy Histogram (Left Panel)
% =========================================================================
subplot('Position', pos1);
ax1 = gca; hold on; box on;

% Determine plotting range
u_all = [U_pt; U_flow];
if isempty(u_all), u_all = 0; end
start_u = min(u_all); end_u = max(u_all);
pts_u = linspace(start_u, end_u, 200);

% Plot PT (Baseline)
if ~isempty(U_pt)
    [f_pt, xi_pt] = ksdensity(U_pt, pts_u);
    % Plot as Line
    plot(xi_pt, f_pt, 'Color', COLOR_PT, 'LineWidth', 2.0, 'DisplayName', 'Parallel Tempering');
end

% Plot Flow (Ours)
if ~isempty(U_flow)
    [f_flow, xi_flow] = ksdensity(U_flow, pts_u, 'Weights', w_flow);
    
    % Fill with transparency
    fill([xi_flow, fliplr(xi_flow)], [f_flow, zeros(size(f_flow))], COLOR_FLOW, ...
         'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xi_flow, f_flow, 'Color', COLOR_FLOW, 'LineWidth', 2.0, 'LineStyle', '--', ...
         'DisplayName', 'Jeffreys Flow');
end

xlabel('Potential Energy U(x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Density \rho(U)', 'FontSize', 12, 'FontWeight', 'bold');
title('Energy Distribution Comparison', 'FontSize', 12);
legend('Location', 'northeast');
axis square; 
grid on;
if ~isempty(u_all), xlim([start_u, -70]); end

% =========================================================================
% 6. Plot 2: 1D Marginal Distribution (Right Panel)
% =========================================================================
subplot('Position', pos2);
ax2 = gca; hold on; box on;

dim_idx = 1; % Plotting x_1
x_min = para.X_LIM_PLOT(1);
x_max = para.X_LIM_PLOT(2);
pts_x = linspace(x_min, x_max, 200);

% Plot PT (Baseline)
if ~isempty(samples_pt)
    [f_pt_x, xi_pt_x] = ksdensity(samples_pt(:, dim_idx), pts_x);
    plot(xi_pt_x, f_pt_x, 'Color', COLOR_PT, 'LineWidth', 2.0, 'DisplayName', 'Parallel Tempering');
end

% Plot Flow (Ours)
if ~isempty(samples_flow)
    [f_flow_x, xi_flow_x] = ksdensity(samples_flow(:, dim_idx), pts_x, 'Weights', w_flow);
    
    fill([xi_flow_x, fliplr(xi_flow_x)], [f_flow_x, zeros(size(f_flow_x))], COLOR_FLOW, ...
         'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xi_flow_x, f_flow_x, 'Color', COLOR_FLOW, 'LineWidth', 2.0, 'LineStyle', '--', ...
         'DisplayName', 'Jeffreys Flow');
end

xlabel(sprintf('Coordinate x_%d', dim_idx), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('Marginal Density p(x_%d)', dim_idx), 'FontSize', 12, 'FontWeight', 'bold');
title('1D Marginal Structure', 'FontSize', 12);
legend('Location', 'northeast');
axis square;
grid on;
xlim([x_min, x_max]);

% =========================================================================
% 7. Final Wrap-up
% =========================================================================
sgtitle(sprintf('Benchmark Comparison for %s', para.NAME), 'FontSize', 16, 'FontWeight', 'bold');

save_name = sprintf('%s_plot_results.pdf', para.ABBR);
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

% =========================================================================
% Local Functions
% =========================================================================

function w_norm = normalize_weights(log_w)
    if isempty(log_w), w_norm=[]; return; end
    % Robust Log-Sum-Exp
    max_log_w = max(log_w);
    w = exp(log_w - max_log_w);
    w_norm = w / sum(w);
end