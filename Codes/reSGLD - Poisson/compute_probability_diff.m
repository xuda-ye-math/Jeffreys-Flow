% compute_probability_diff.m
% Calculates the probability distribution across the 24 modes for MU and the 3 NU distributions
% and computes the total variation distance to the uniform distribution.

clear; clc; close all;

% =========================================================================
% 1. Load Configurations and True Parameters
% =========================================================================
addpath(genpath('./data'));
para_path = './data/para.mat';
if ~exist(para_path, 'file')
    error('Configurations para.mat not found. Please run parameters.m first.');
end
load(para_path, 'para'); 

% True parameters: 8x1 vector, consisting of 4 sources (x, y)
theta_true = para.theta_true(:)'; % 1x8
sources_true = reshape(theta_true, 2, 4)'; % 4x2 matrix, each row is a source
num_sources = 4;

% Generate all 24 permutations of [1, 2, 3, 4]
P = perms(1:num_sources); % 24x4 matrix
num_modes = size(P, 1);   % 24

% Create 24 reference points in R^8 by permutating the 4 sources
ref_modes = zeros(num_modes, 8);
for i = 1:num_modes
    temp_sources = sources_true(P(i, :), :); % 4x2 matrix of permuted sources
    temp_row = reshape(temp_sources', 1, 8); % Flatten back to 1x8
    ref_modes(i, :) = temp_row;
end

% =========================================================================
% 2. Setup Data Loading
% =========================================================================
data_dir = './data';
k = para.M; % Extract index from max beta

% Define configs for the 4 distributions
configs = struct();
configs(1).name = sprintf('samples_MU_%d.mat', k);
configs(1).title = 'MU (SGLD)';
configs(1).weighted = false;

configs(2).name = sprintf('samples_e_NU_%d.mat', k);
configs(2).wname = sprintf('weights_e_NU_%d.mat', k);
configs(2).title = 'e_NU (Exact)';
configs(2).weighted = true;

configs(3).name = sprintf('samples_f_NU_%d.mat', k);
configs(3).wname = sprintf('weights_f_NU_%d.mat', k);
configs(3).title = 'f_NU (Full)';
configs(3).weighted = true;

configs(4).name = sprintf('samples_s_NU_%d.mat', k);
configs(4).wname = sprintf('weights_s_NU_%d.mat', k);
configs(4).title = 's_NU (Stochastic)';
configs(4).weighted = true;

% Uniform distribution over 24 modes
prob_uniform = ones(1, num_modes) / num_modes;

% Store probability differences (Total Variation and Total Variance)
prob_diffs_tv = zeros(1, 4);

fprintf('--- Calculating Probability Differences over %d modes ---\n', num_modes);

for i = 1:4
    cfg = configs(i);
    fpath = fullfile(data_dir, cfg.name);
    
    if ~exist(fpath, 'file')
         fprintf('Warning: File %s not found.\n', cfg.name);
         prob_diffs_tv(i) = NaN;
         continue;
    end
    
    tmp = load(fpath);
    if isfield(tmp, 'samples')
        samples = tmp.samples;
    else
        vars = fieldnames(tmp);
        samples = tmp.(vars{1});
    end
    
    if isempty(samples)
        fprintf('Warning: File %s has no samples.\n', cfg.name);
        prob_diffs_tv(i) = NaN;
        continue;
    end
    
    N = size(samples, 1);
    
    % Weights
    log_w = [];
    if cfg.weighted
        wpath = fullfile(data_dir, cfg.wname);
        if exist(wpath, 'file')
            tmp_w = load(wpath);
            if isfield(tmp_w, 'weights')
                log_w = tmp_w.weights;
            else
                vars_w = fieldnames(tmp_w);
                log_w = tmp_w.(vars_w{1});
            end
        end
    end
    
    if cfg.weighted && ~isempty(log_w)
        w_norm = normalize_weights(log_w);
    else
        w_norm = ones(N, 1) / N;
    end
    
    % =========================================================================
    % 3. Mode Assignment & Probability Calculation
    % =========================================================================
    % Compute pairwise Euclidean distances from samples to ref_modes
    try
        dist_matrix = pdist2(double(samples), ref_modes, 'euclidean');
        [~, mode_indices] = min(dist_matrix, [], 2);
    catch err
        % Fallback if pdist2 is not available or faces memory issues
        fprintf('Using fallback for distance calculation...\n');
        mode_indices = zeros(N, 1);
        for m = 1:N
            dists = sum((ref_modes - samples(m, :)).^2, 2);
            [~, mode_indices(m)] = min(dists);
        end
    end
    
    % Aggregate probabilities for each mode
    prob_dist = zeros(1, num_modes);
    for m = 1:num_modes
        prob_dist(m) = sum(w_norm(mode_indices == m));
    end
    
    % Calculate Total Variation Distance: 0.5 * sum(|P - Q|)
    % Also calculate sum of squared differences just in case
    total_variation = 0.5 * sum(abs(prob_dist - prob_uniform));
    total_variance = sum((prob_dist - prob_uniform).^2);
    
    prob_diffs_tv(i) = total_variation;
    
    fprintf('%-20s: Total Variation = %.4f (Sum of Sq Diff = %.4f)\n', cfg.title, total_variation, total_variance);
end

% Final result array returned (4 numbers, probability difference for each method)
fprintf('\nFinal probability difference (Total Variation) for [MU, e_NU, f_NU, s_NU]:\n');
disp(prob_diffs_tv);

% Save to a resulting variable diffs for output
probability_differences = prob_diffs_tv;

% =========================================================================
% Local Functions
% =========================================================================

function w_norm = normalize_weights(log_w)
    if isempty(log_w)
        w_norm = []; 
        return; 
    end
    log_w = double(log_w(:)); 
    w = exp(log_w - max(log_w)); 
    w_norm = w / sum(w);
end
