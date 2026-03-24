clear; clc; rng(28);

% =========================================================================
% 0. Core Configuration
% =========================================================================
N_CORES = 8;         % Number of CPU cores
para = Nonlinear_8D();   % Initialize Parameter Object (Twisted Nonlinear Rastrigin)

fprintf('Initializing Parallel Script for %s (%s)...\n', para.NAME, para.ABBR);
para.disp_info();

% ---- [NEW] Hybrid Ladder Logic ----
% Re-calculate BETA_LIST to satisfy step size constraints (Max step = 0.1)
% Strategy: 
%   - If Rate > 1: Arithmetic near 1, Geometric near 0.
%   - If Rate < 1: Decaying Geometric from Min to 1 (Steps shrink).
%   - If Rate = 1: Pure Arithmetic.
fprintf('[Ladder] Re-computing Beta List with max step %.3f...\n', para.DELTA_MAX);
new_betas = get_hybrid_ladder(para.BETA_MIN, para.BETA_RATE, para.DELTA_MAX);
para.BETA_LIST = new_betas;
para.PT_M = length(new_betas) - 1; % Update step count
% -----------------------------------

% Directory Setup
data_dir = './data';
if ~exist(data_dir, 'dir'); mkdir(data_dir); end

% Save BETA_LIST (Lambda Ladder) for Python Interface
beta_file_name = sprintf('%s_BETA_LIST.mat', para.ABBR);
beta_save_path = fullfile(data_dir, beta_file_name);
BETA_LIST = para.BETA_LIST; 
save(beta_save_path, 'BETA_LIST');
fprintf('[Config] Saved Lambda/Beta List to %s\n', beta_file_name);

% =========================================================================
% 1. Setup Parallel Environment & DataQueue
% =========================================================================

% Check/Start Parallel Pool
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= N_CORES
    delete(gcp('nocreate'));
    parpool(N_CORES);
end

% Setup DataQueue for Progress Monitoring
q = parallel.pool.DataQueue;
afterEach(q, @display_progress_callback);

% =========================================================================
% 2. Simulation Logic
% =========================================================================

fprintf('Current BETA_LIST for Parallel Tempering:\n')
disp(para.BETA_LIST)

% Number of replicas (Total Steps in Ladder)
N_REPLICAS = length(para.BETA_LIST);
fprintf('Detected %d Replicas (Lambdas) from parameter config.\n', N_REPLICAS);

% --- Check for Existing Files Loop ---
all_files_exist = true;
fprintf('[Check] Verifying existence of %d sample files...\n', N_REPLICAS);

for k = 0 : N_REPLICAS - 1
    fname = sprintf('%s_samples_MU_%d.mat', para.ABBR, k);
    fpath = fullfile(data_dir, fname);
    
    if ~exist(fpath, 'file')
        all_files_exist = false;
        fprintf('   -> Missing: %s\n', fname);
        break; 
    end
end

if all_files_exist
    fprintf('[Check] All %d datasets found. Skipping regeneration.\n', N_REPLICAS);
    
    % Load target for analysis check
    target_idx = N_REPLICAS - 1;
    target_file = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, target_idx));
    loaded_struct = load(target_file);
    target_var_name = sprintf('samples_MU_%d', target_idx);
    
    if isfield(loaded_struct, target_var_name)
        samples_target = loaded_struct.(target_var_name);
    end
    
else
    fprintf('[Check] Data incomplete or mismatch. Starting regeneration...\n');

    % --- Part I: Base Distribution (Serial) ---
    fprintf('\n[Sampling MU_0] Generating Base Samples (%d)...\n', para.MU_SIZE);
    
    save_path_0 = fullfile(data_dir, sprintf('%s_samples_MU_0.mat', para.ABBR));
    
    % Generate samples from the exact Base Potential defined in the class
    % We use simple Gaussian sampling because the base is defined as Gaussian
    % consistent with compute_potential_mixed(x, 0).
    mu_base = zeros(1, para.DIM) + para.MEAN; 
    sigma_mat = eye(para.DIM) * (para.SIGMA^2);
    
    samples_MU_0 = mvnrnd(mu_base, sigma_mat, para.MU_SIZE);
    save(save_path_0, 'samples_MU_0');
    
    % --- Part II: Parallel Tempering (Parallel) ---
    total_samples = para.MU_SIZE;
    samples_per_core = ceil(total_samples / N_CORES);
    
    fprintf('\n[PT-Parallel] Starting Main Sampling on %d Cores (Monitoring Core 1)...\n', N_CORES);
    
    % Pre-allocate cells for results
    results_all = cell(N_CORES, 1);
    
    global_tic = tic;
    
    parfor k = 1:N_CORES
        % Local copies
        local_para = para;
        local_samples_per_core = samples_per_core;
        
        % Local Storage: [Samples, Replicas, Dimension]
        local_res = zeros(local_samples_per_core, N_REPLICAS, local_para.DIM);
        
        % Local Stats
        swap_attempts = zeros(N_REPLICAS - 1, 1);
        swap_accepts  = zeros(N_REPLICAS - 1, 1);
        
        % Initialize Replicas Locally (Start from Base Distribution)
        l_mu = zeros(1, local_para.DIM) + local_para.MEAN;
        l_sig = eye(local_para.DIM) * (local_para.SIGMA^2);
        x = mvnrnd(l_mu, l_sig, N_REPLICAS);
        
        % Local Parameters
        steps_per_samp = round(local_para.COR_TIME / local_para.DT);
        n_warmup = ceil(local_para.WARM_TIME / local_para.COR_TIME);
        glob_step = 0;
        
        core_tic = tic;
        
        % --- Warm-up ---
        for ep = 1:n_warmup
            for s = 1:steps_per_samp
                glob_step = glob_step + 1;
                x = local_para.integrator(x);
                if mod(glob_step, local_para.SWAP_INTERVAL_STEPS) == 0
                    x = perform_swap(x, local_para, rand() > 0.5);
                end
            end
        end
        
        % --- Sampling ---
        for i = 1:local_samples_per_core
            for s = 1:steps_per_samp
                glob_step = glob_step + 1;
                x = local_para.integrator(x);
                
                if mod(glob_step, local_para.SWAP_INTERVAL_STEPS) == 0
                    swap_cnt = glob_step / local_para.SWAP_INTERVAL_STEPS;
                    is_even = mod(swap_cnt, 2) == 0;
                    [x, acc, att] = perform_swap(x, local_para, is_even);
                    
                    swap_attempts = swap_attempts + att;
                    swap_accepts  = swap_accepts + acc;
                end
            end
            
            local_res(i, :, :) = x;
            
            % --- MONITORING (Core 1 Only) ---
            if k == 1
                if mod(i, 1000) == 0 || i == local_samples_per_core
                    safe_att = swap_attempts; safe_att(safe_att==0) = 1;
                    rates = swap_accepts ./ safe_att;
                    min_rate = min(rates);
                    
                    info = struct();
                    info.iter = i;
                    info.total = local_samples_per_core;
                    info.min_rate = min_rate;
                    info.elapsed = toc(core_tic);
                    send(q, info);
                end
            end
        end
        
        results_all{k} = local_res;
    end
    
    total_time = toc(global_tic);
    fprintf('[PT-Parallel] Finished in %.1fs.\n', total_time);
    
    % =========================================================================
    % 3. Save Data
    % =========================================================================
    fprintf('\n[Saving] Processing and saving %d datasets...\n', N_REPLICAS);
    
    full_data = vertcat(results_all{:}); 
    
    if size(full_data, 1) > para.MU_SIZE
        full_data = full_data(1:para.MU_SIZE, :, :);
    end
    
    for k = 1:N_REPLICAS
        samples_k = squeeze(full_data(:, k, :));
        
        file_idx = k - 1; 
        var_name = sprintf('samples_MU_%d', file_idx);
        file_name = sprintf('%s_%s.mat', para.ABBR, var_name);
        file_path = fullfile(data_dir, file_name);
        
        S = struct();
        S.(var_name) = samples_k;
        save(file_path, '-struct', 'S');
        
        if k == 1 || k == N_REPLICAS || mod(k, 10) == 0
            fprintf('  -> Saved %s (%d samples)\n', file_name, size(samples_k, 1));
        end
    end
    
    samples_target = squeeze(full_data(:, end, :));
end

% =========================================================================
% 4. Analysis
% =========================================================================
if exist('compute_bias', 'file') && exist('samples_target', 'var')
    compute_bias(para, samples_target, 1.0);
end


% =========================================================================
% Local Functions
% =========================================================================

function display_progress_callback(data)
    fprintf('  [Core 1] Progress: %5d / %d | Min Swap: %.1f%% | Time: %.1fs\n', ...
        data.iter, data.total, data.min_rate * 100, data.elapsed);
end

function [x, acc_vec, att_vec] = perform_swap(x, para, is_even_phase)
    % Perform Swap for Linear Interpolation: U_lam = (1-lam)U0 + lam*U1
    lambdas = para.BETA_LIST; 
    M = length(lambdas);
    acc_vec = zeros(M - 1, 1);
    att_vec = zeros(M - 1, 1);
    
    % 1. Compute Potentials using Interface
    U_base_all = para.compute_potential_mixed(x, 0.0);   % [M, 1]
    U_target_all = para.compute_potential_mixed(x, 1.0); % [M, 1]
    
    % 2. Define Pairs
    if is_even_phase
        pairs = 1:2:(M - 1);
    else
        pairs = 2:2:(M - 1);
    end
    
    for k = pairs
        idx_A = k;
        idx_B = k + 1;
        att_vec(k) = 1;
        
        lam_A = lambdas(idx_A);
        lam_B = lambdas(idx_B);
        
        % Linear Interpolation Delta Logic
        Diff_A = U_target_all(idx_A) - U_base_all(idx_A);
        Diff_B = U_target_all(idx_B) - U_base_all(idx_B);
        
        delta = (lam_A - lam_B) * (Diff_B - Diff_A);
        
        if delta <= 0 || rand() < exp(-delta)
            % Swap Coordinates
            tmp_x = x(idx_A, :);
            x(idx_A, :) = x(idx_B, :);
            x(idx_B, :) = tmp_x;
            acc_vec(k) = 1;
        end
    end
end

function beta_list = get_hybrid_ladder(beta_min, beta_rate, delta_max)
    % GET_HYBRID_LADDER Generates an interpolation ladder between beta_min and 1.0
    %
    % Input:
    %   beta_min  - Smallest non-zero beta
    %   beta_rate - Geometric growth/decay rate.
    %   delta_max - Maximum allowed arithmetic step size.
    % Output:
    %   beta_list - Column vector [0; ...; 1]
    
    if abs(beta_rate - 1.0) < 1e-6
        % ---- CASE A: Pure Arithmetic (Rate = 1) ----
        % Equal steps of size at most delta_max
        dist = 1.0 - beta_min;
        n_steps = ceil(dist / delta_max);
        
        seq = linspace(beta_min, 1.0, n_steps + 1);
        beta_list = [0, seq]';
        
    elseif beta_rate > 1.0
        % ---- CASE B: Standard Hybrid (Rate > 1) ----
        % Geometric growth from min, switching to Arithmetic max-step near 1.
        
        % 1. Determine Threshold for switch
        % Condition: step / beta <= rate - 1 => beta >= step / (rate - 1)
        threshold = delta_max / (beta_rate - 1);
        
        % 2. Generate Arithmetic Part (Backwards from 1.0)
        arith_desc = [];
        curr = 1.0;
        while curr > threshold + 1e-9
            arith_desc(end+1) = curr; %#ok<AGROW>
            curr = curr - delta_max;
        end
        beta_top = curr;
        
        % 3. Generate Geometric Part (From beta_min to beta_top)
        if beta_top <= beta_min
             geom_part = beta_min; 
        else
            ratio = beta_top / beta_min;
            n_steps = ceil(log(ratio) / log(beta_rate));
            actual_rate = ratio ^ (1 / n_steps);
            geom_part = beta_min * (actual_rate .^ (0 : n_steps-1));
        end
        
        % 4. Combine: 0, Geometric, beta_top, Arithmetic Ascending
        arith_asc = [beta_top, flip(arith_desc)];
        beta_list = [0, geom_part, arith_asc]';
        
    else
        % ---- CASE C: Decaying Geometric (Rate < 1) ----
        % Step sizes decrease geometrically as beta approaches 1.
        % This places high density near 1.0 (Target).
        %
        % Logic:
        %   delta_{k+1} = delta_k * rate
        %   Sum(delta) = 1.0 - beta_min
        %   Constraint: delta_0 <= delta_max
        
        dist = 1.0 - beta_min;
        
        % 1. Check Feasibility of Infinite Sum
        % Max possible distance with delta_max: S_inf = delta_max / (1 - rate)
        max_dist_capacity = delta_max / (1.0 - beta_rate);
        
        if max_dist_capacity < dist
            % Warning: Rate is too small to cover distance even with max step.
            % Must violate delta_max or just stretch to fit.
            % We choose to find an N that makes steps small enough at the end.
            % (e.g., last step factor approx 1e-6)
            N = ceil(log(1e-6) / log(beta_rate));
        else
            % Feasible: Find min N such that start_step <= delta_max
            % delta_0 = dist * (1-r) / (1-r^N) <= delta_max
            % => r^N <= 1 - (dist*(1-r)/delta_max)
            rhs = 1.0 - (dist * (1.0 - beta_rate)) / delta_max;
            N = ceil(log(rhs) / log(beta_rate));
        end
        
        N = max(N, 1);
        
        % 2. Recompute exact initial step to hit 1.0 exactly in N steps
        delta_0 = dist * (1.0 - beta_rate) / (1.0 - beta_rate^N);
        
        % 3. Generate Sequence
        % beta_k = beta_min + delta_0 * (1 - r^k) / (1 - r)
        k_vec = 0:N;
        % Geometric series sum formula terms: (1 - r^k)/(1 - r)
        geom_sum_factors = (1.0 - beta_rate.^k_vec) ./ (1.0 - beta_rate);
        beta_seq = beta_min + delta_0 .* geom_sum_factors;
        
        % Numerical clamping
        beta_seq(1) = beta_min;
        beta_seq(end) = 1.0;
        
        beta_list = [0; beta_seq'];
    end
end