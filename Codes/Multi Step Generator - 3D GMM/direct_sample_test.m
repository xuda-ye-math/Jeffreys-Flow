clear; clc; close all; rng(27);

% =========================================================================
% 1. Initialization
% =========================================================================

% Initialize the 3D GMM Problem Object
para = GMM_3D();
fprintf('Initializing Direct Sampling for %s...\n', para.NAME);
para.disp_info();

% Directory Setup
fig_dir = './figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Parameters
N_samples = para.MU_SIZE;

% =========================================================================
% 2. Direct Sampling from GMM
% =========================================================================
fprintf('\n[Sampling] Generating %d samples directly from the mixture...\n', N_samples);

% 1. Retrieve Parameters
weights_log = para.TARGET_LOG_WEIGHTS;
weights = exp(weights_log);
weights = weights / sum(weights); % Ensure normalized

means = para.TARGET_MEANS;         % [K, 3]
precisions = para.TARGET_PRECISIONS; % [K, 3, 3]

K = length(weights);
samples = zeros(N_samples, para.DIM);

% 2. Sample Component Indices
% randsample uses probabilities to pick indices 1..K
comp_indices = randsample(1:K, N_samples, true, weights);

% 3. Sample from Gaussian Components
% We count how many samples fall into each component to vectorize mvnrnd
counts = histcounts(comp_indices, 1:K+1);

current_idx = 1;
for k = 1:K
    n_k = counts(k);
    if n_k == 0, continue; end
    
    mu_k = means(k, :);
    prec_k = squeeze(precisions(k, :, :));
    
    % Covariance = inv(Precision)
    sigma_k = inv(prec_k);
    
    % Generate n_k samples from N(mu_k, sigma_k)
    % mvnrnd(mu, sigma, n) returns [n, d]
    x_k = mvnrnd(mu_k, sigma_k, n_k);
    
    % Store
    samples(current_idx : current_idx + n_k - 1, :) = x_k;
    current_idx = current_idx + n_k;
end

% Shuffle samples (optional, to mix components in the array)
samples = samples(randperm(N_samples), :);

fprintf('[Sampling] Done.\n');

% =========================================================================
% 3. Visualization (2D Projection)
% =========================================================================
fprintf('\n[Plotting] Generating 2D projection (X1-X2)...\n');

f = figure('Name', ['Direct Sampling - ' para.NAME], ...
           'Color', 'w', 'Position', [100, 100, 800, 700]);

hold on; box on;

% Scatter Plot
% X1 vs X2 (Projecting out X3)
scatter(samples(:, 1), samples(:, 2), 5, [0.2, 0.4, 0.8], 'filled', ...
    'MarkerFaceAlpha', 0.5);

% Formatting
xlabel('x_1'); 
ylabel('x_2');
title(sprintf('\\bf Direct Sampling from %s (N=%d)\n2D Projection (x_1, x_2)', ...
    para.NAME, N_samples), 'Interpreter', 'tex', 'FontSize', 14);

xlim(para.X_LIM_PLOT);
ylim(para.Y_LIM_PLOT);
axis square;
grid on;

set(gca, 'FontSize', 12, 'LineWidth', 1.0, 'Layer', 'top');

% =========================================================================
% 4. Save Figure
% =========================================================================
save_path = fullfile(fig_dir, 'direct_sample.png');
fprintf('Saving figure to %s (300 DPI)...\n', save_path);
exportgraphics(f, save_path, 'Resolution', 300);

fprintf('\n[Done] Script finished successfully.\n');