clear; clc; close all;

% =========================================================================
% 1. Configuration & Initialization
% =========================================================================
addpath('./potential');
    
% Instantiate Solvated_16D to get parameters
try
    para = Solvated_16D();
catch
    % Fallback if class not found in path
    warning('Solvated_16D class not found. Using hardcoded parameters.');
    para.ABBR = 'SL';
    para.NAME = 'Solvated 16D';
    para.A_GRID = 2.0;
    para.FREQ_GRID = 4.0;
    para.X_PERI_LIM = [-pi, pi];
end

fprintf('Generating 1D PMF Plot for Step 31 %s (%s)...\n', para.NAME, para.ABBR);

data_dir = './data';
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% =========================================================================
% 2. Visualization Setup
% =========================================================================

% Target Step: 31 (Beta = 1.0)
target_step = 31;
fprintf('Plotting ONLY Step %d (Beta=1.0)\n', target_step);

% Figure Setup
fig_w = 550; 
fig_h = 450; 
f = figure('Name', '1D PMF Comparison (Step 31)', 'Color', 'w', 'Position', [100, 100, fig_w, fig_h]);

% Tuning Layout
marg_l = 0.12;
marg_r = 0.05;
marg_b = 0.12;
marg_t = 0.10;

% Colors
COLOR_PT   = [0.2, 0.2, 0.2];     % Black/Grey (Baseline - PT)
COLOR_FLOW = [0.85, 0.1, 0.1];    % Red (Ours - Flow)
COLOR_GT   = [0.0, 0.6, 0.2];     % Green (Ground Truth)

% =========================================================================
% 3. Load Data
% =========================================================================

k = target_step;
fname_mu = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, k));
fname_nu = fullfile(data_dir, sprintf('%s_samples_NU_%d.mat', para.ABBR, k));
fname_w  = fullfile(data_dir, sprintf('%s_weights_NU_%d.mat', para.ABBR, k));

samples_pt = load_samples(fname_mu);
samples_flow = load_samples(fname_nu);
log_w = load_weights(fname_w);

% Normalize weights
w_flow = [];
if ~isempty(log_w)
    w_flow = normalize_weights(log_w);
else
    if ~isempty(samples_flow)
        w_flow = ones(size(samples_flow,1), 1) / size(samples_flow,1);
    end
end

% Function to wrap x to [-pi, pi]
wrap_fn = @(x) mod(x + pi, 2*pi) - pi;

if ~isempty(samples_pt), samples_pt(:,1) = wrap_fn(samples_pt(:,1)); end
if ~isempty(samples_flow), samples_flow(:,1) = wrap_fn(samples_flow(:,1)); end

% =========================================================================
% 4. Compute PMF (Free Energy F(x) = -ln p(x))
% =========================================================================

% Grid for PMF
x_grid = linspace(-pi, pi, 100);

% A. Ground Truth (Theoretical)
% U_eff(x1) = A * cos(4*x1) + const
% We only care about the shape, so we plot U(x1) directly
U_gt = para.A_GRID * cos(para.FREQ_GRID * x_grid);
% Align mean for comparison
U_gt = U_gt - mean(U_gt); 

% B. PMF for PT
pmf_pt = [];
if ~isempty(samples_pt)
    % Compute periodic density
    p_pt = periodic_density(samples_pt(:,1), x_grid, [], 0.15);
    
    % F = -ln(p)
    % Avoid log(0)
    p_pt(p_pt < 1e-10) = 1e-10;
    pmf_pt = -log(p_pt);
    % Align mean to GT for overlap
    pmf_pt = pmf_pt - mean(pmf_pt);
end

% C. PMF for Flow
pmf_flow = [];
if ~isempty(samples_flow)
    % Compute periodic density with weights
    p_flow = periodic_density(samples_flow(:,1), x_grid, w_flow(:), 0.15);
    
    p_flow(p_flow < 1e-10) = 1e-10;
    pmf_flow = -log(p_flow);
    % Align mean to GT for overlap
    pmf_flow = pmf_flow - mean(pmf_flow);
end


% =========================================================================
% 5. Plot
% =========================================================================
ax = axes('Position', [marg_l, marg_b, 1-marg_l-marg_r, 1-marg_b-marg_t]);
hold(ax, 'on'); box(ax, 'on');

% 1. Plot Theoretical (Ground Truth)
plot(x_grid, U_gt, '--', 'Color', COLOR_GT, 'LineWidth', 2.0, 'DisplayName', 'Theoretical');

% 2. Plot PT
if ~isempty(pmf_pt)
    plot(x_grid, pmf_pt, '-', 'Color', COLOR_PT, 'LineWidth', 1.8, 'DisplayName', 'Parallel Tempering');
end

% 3. Plot Flow
if ~isempty(pmf_flow)
    plot(x_grid, pmf_flow, '-.', 'Color', COLOR_FLOW, 'LineWidth', 2.0, 'DisplayName', 'Jeffreys Flow');
end

% Styling
xlim([-pi, pi]);
ylim([-2.3,2.9])
xlabel('Coordinate x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Free Energy F(x_1) (Shifted)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('PMF for 16D %s', para.NAME), 'FontSize', 14, 'FontWeight', 'bold');

legend('Location','north','FontSize', 9);
grid on;
axis square;

% =========================================================================
% 6. Save
% =========================================================================

save_name = sprintf('%s_plot_results_2.pdf', para.ABBR);
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
            fnames = fieldnames(tmp);
            samples = tmp.(fnames{1});
        end
        % Filter only
        samples = samples(all(isfinite(samples), 2), :);
    else
        warning('File not found: %s', fname);
    end
end

function w = load_weights(fname)
    w = [];
    if exist(fname, 'file')
        tmp = load(fname);
        if isfield(tmp, 'weights')
            w = tmp.weights;
        else
            fnames = fieldnames(tmp);
            w = tmp.(fnames{1});
        end
        w = w(:);
    end
end

function w_norm = normalize_weights(log_w)
    if isempty(log_w), w_norm=[]; return; end
    max_log_w = max(log_w);
    w = exp(log_w - max_log_w);
    w_norm = w / sum(w);
end

function p = periodic_density(samples, grid_pts, weights, bw)
    % Augment samples to handle periodic boundary [-pi, pi]
    % Assumes period is 2*pi
    
    L = 2*pi;
    
    % Shifted copies
    s_left  = samples - L;
    s_right = samples + L;
    
    s_aug = [s_left; samples; s_right];
    
    if ~isempty(weights)
        w_aug = [weights; weights; weights];
        % Normalize again just in case, though ksdensity usually handles relative weights
        w_aug = w_aug / sum(w_aug);
        
        [p_aug, xi_aug] = ksdensity(s_aug, grid_pts, 'Bandwidth', bw, 'Weights', w_aug);
    else
        [p_aug, xi_aug] = ksdensity(s_aug, grid_pts, 'Bandwidth', bw);
    end
    
    % The density from ksdensity on augmented data will be 1/3 of the true density 
    % (since we tripled the mass but kept the domain same-ish effectively for normalization over R).
    % Alternatively, we just computed p(x) around the center.
    % Actually, if we use 'Weights' that sum to 1 over the augmented set, the integral over R is 1.
    % The integral over [-pi, pi] for the center part should be roughly 1/3.
    % We want the density to integrate to 1 over [-pi, pi].
    
    % Let's correct this:
    % p_aug is the density of the *augmented* distribution.
    % The augmented distribution has 3 modes (left, center, right).
    % We only care about the center one, but we want it to be the full probability mass.
    % So we need to multiply by 3.
    
    p = p_aug * 3;
end
