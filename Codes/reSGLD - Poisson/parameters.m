% parameters.m
% 2D Screened Poisson Benchmark Parameter Configuration
%
% This script defines all the physical, numerical, and algorithmic parameters
% required for the 2D Source Inversion (SGLD) benchmark. It also computes
% the synthetic ground truth observations and saves everything into a 
% 'para.mat' file inside the 'data' directory for other scripts to load.

clear; clc; close all;
rng(27); % Strict random seed for exact reproducibility

fprintf('--- 2D Screened Poisson Benchmark Configuration ---\n');

% =========================================================================
% 1. Define Master Parameters Structure
% =========================================================================
para = struct();

% --- 1.1 Grid Setup --- 
para.N_cells = 40;     % 40 cells means h = 1 / 40 = 0.025. 
                       % This drastically speeds up solving A \ v by reducing A to 1600x1600 instead of 10000x10000.
para.h = 1.0 / para.N_cells;
para.dim = 8;          % 4 sources * 2D coordinates (x, y) = 8 parameters

% --- 1.2 Physical Constants ---
para.alpha = 1;      % Screening/absorption coefficient
para.c = 1.0;         % Source intensities (fixed for all 4 sources)
para.gamma = 0.1;     % Source width (standard deviation)

% --- 1.3 Ground Truth Parameters ---
% 4 sources placed symmetrically with slight offsets
para.theta_true = [0.22; 0.28; ...  % Source 1 (Bottom-Left)
                   0.28; 0.72; ...  % Source 2 (Top-Left)
                   0.72; 0.22; ...  % Source 3 (Bottom-Right)
                   0.78; 0.78];     % Source 4 (Top-Right)

% --- 1.4 SGLD & Parallel Tempering Parameters ---
para.batch_size = 9;        % Batch size for SGLD (using 9 specific sensors)
para.sigma_noise = 0.02;   % Standard deviation of observation noise
para.total_obs   = 81;          % Sparse 9x9 uniform grid of sensors

% Scale Total Samples
para.total_samples = 100000; % number of samples per chain
para.beta_list = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]; % temperature ladder
para.M = length(para.beta_list) - 1; % M+1 Temperature chains (beta_0 to beta_M)
para.dt = 2e-5;              % SGLD Step size

para.swap_interval_steps = 10;  % Propose a temperature swap every ... steps
para.record_interval = 40;     % Only record a sample every ... steps  

% =========================================================================
% 2. Generate Synthetic Observations
% =========================================================================
fprintf('Initializing Forward Solver to generate synthetic data...\n');
solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);

% Solve for the true state field
u_true_mat = solver.solve_forward_mat(para.theta_true);

% Extract observations at the 9x9 sensor network
[u_obs_clean_mat, obs_indices] = get_u_obs_mat(u_true_mat, para.N_cells);

% Store the exact linear indices of the sensors for gradient computation
para.obs_indices = obs_indices;

% Flatten clean observations
y_clean = u_obs_clean_mat(:);

% Add Gaussian white noise
noise = para.sigma_noise * randn(size(y_clean));
y_obs = y_clean + noise;

% Store observations in the parameter struct
para.y_obs = y_obs;

fprintf('Synthetic observations generated. Max SNR roughly: %.2f\n', max(abs(y_clean)) / para.sigma_noise);

% =========================================================================
% 3. Save Configuration to File
% =========================================================================
data_dir = './data';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf('Created data directory: %s\n', data_dir);
end

save_path = fullfile(data_dir, 'para.mat');
save(save_path, 'para');
fprintf('Successfully saved benchmark configuration to %s\n', save_path);

% Plot the generated data field for verification
figure('Name', 'Synthetic Ground Truth & Sensors');
contourf(solver.X, solver.Y, u_true_mat, 50, 'LineStyle', 'none');
colormap parula; hold on; colorbar;
title('Synthetic Ground Truth Field & Sensors');

% Sensors
[obs_X, obs_Y] = meshgrid(0.1:0.1:0.9, 0.1:0.1:0.9);
plot(obs_X(:), obs_Y(:), 'ks', 'MarkerSize', 6, 'MarkerFaceColor', 'k');

% Sources
num_sources = length(para.theta_true) / 2;
for k=1:num_sources
    plot(para.theta_true(2*k-1), para.theta_true(2*k), 'w+', 'MarkerSize', 10, 'LineWidth', 2);
    plot(para.theta_true(2*k-1), para.theta_true(2*k), 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
end
hold off;
axis equal tight;
