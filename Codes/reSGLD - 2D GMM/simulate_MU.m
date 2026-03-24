clear; clc; rng(28);

% =========================================================================
% 0. Core Configuration
% =========================================================================
N_CORES = 4;       % Number of CPU cores
para = GMM_2D();   % Initialize Parameter Object (Updated for Linear Interp)

fprintf('Initializing Parallel Script for %s (%s)...\n', para.NAME, para.ABBR);
para.disp_info();

% Directory Setup
data_dir = './data';
if ~exist(data_dir, 'dir'); mkdir(data_dir); end

% [NEW] Save BETA_LIST (Lambda Ladder) for Python Interface
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

% Number of replicas (Total Steps in Ladder)
N_REPLICAS = length(para.BETA_LIST);
fprintf('Detected %d Replicas (Lambdas) from parameter config.\n', N_REPLICAS);

% --- [UPDATED] Check for Existing Files Loop ---
all_files_exist = true;
fprintf('[Check] Verifying existence of %d sample files (Correct & Naive)...\n', N_REPLICAS);

% Check Correct Files
for k = 0 : N_REPLICAS - 1
    fname = sprintf('%s_samples_correct_MU_%d.mat', para.ABBR, k);
    fpath = fullfile(data_dir, fname);
    if ~exist(fpath, 'file')
        all_files_exist = false; 
        break; 
    end
end

% Check Naive Files
if all_files_exist
    for k = 0 : N_REPLICAS - 1
        fname = sprintf('%s_samples_naive_MU_%d.mat', para.ABBR, k);
        fpath = fullfile(data_dir, fname);
        if ~exist(fpath, 'file')
            all_files_exist = false; 
            break; 
        end
    end
end

if all_files_exist
    fprintf('[Check] All datasets (Correct & Naive) found. Skipping regeneration.\n');
    
    % Load the final target file just to have variable 'samples_MU_M' available for analysis check
    % We load the 'correct' one as default for analysis
    target_idx = N_REPLICAS - 1;
    target_file = fullfile(data_dir, sprintf('%s_samples_correct_MU_%d.mat', para.ABBR, target_idx));
    
    % Load dynamically
    loaded_struct = load(target_file);
    % Variable name inside might still be generic or specific, let's assume we maintain generic internal names
    % But wait, if I save them with specific names, I should check how I save them.
    % I will save them as generic fields 'samples' usually, but let's stick to the struct field pattern used before.
    % To be safe, I'll update the saving logic to use consistent field names or just load whatever is there.
    
    % For now just skipping the detailed load check or keeping it simple
    if isfield(loaded_struct, 'samples')
        samples_target = loaded_struct.samples;
    elseif isfield(loaded_struct, sprintf('samples_MU_%d', target_idx))
        samples_target = loaded_struct.(sprintf('samples_MU_%d', target_idx));
    end
    
else
    fprintf('[Check] Data incomplete or mismatch. Starting regeneration...\n');

    % --- Part I: Base Distribution (Serial) ---
    fprintf('\n[Sampling MU_0] Generating Gaussian Base Samples (%d)...\n', para.MU_SIZE);
    
    % MU_0 is always the first file (index 0)
    save_path_0 = fullfile(data_dir, sprintf('%s_samples_MU_0.mat', para.ABBR));
    
    mu_base = para.MEAN * ones(1, para.DIM);
    sigma_base = para.SIGMA^2 * eye(para.DIM);
    samples_MU_0 = mvnrnd(mu_base, sigma_base, para.MU_SIZE);
    save(save_path_0, 'samples_MU_0');
    
    % --- Part II: Parallel Tempering (Parallel) ---
    total_samples = para.MU_SIZE;
    samples_per_core = ceil(total_samples / N_CORES);
    
    fprintf('\n[PT-Parallel] Starting Main Sampling on %d Cores (Monitoring Core 1)...\n', N_CORES);
    
    % Pre-allocate cells for results (One cell per core)
    results_all = cell(N_CORES, 1);
    
    % Store Warm-up States to continue simulation
    warmup_states = cell(N_CORES, 1);
    warmup_stats  = cell(N_CORES, 1);
    
    % Start Global Timer
    global_tic = tic;
    
    % --- PHASE 1: WARM-UP & SIGMA ESTIMATION ---
    fprintf('\n[PT-Parallel] Phase 1: Warm-up & Variance Estimation...\n');
    parfor k = 1:N_CORES
        % Local copies
        local_para = para;
        
        % Initialize PT Ladder Locally
        l_mu = local_para.MEAN * ones(1, local_para.DIM);
        l_sig = local_para.SIGMA * eye(local_para.DIM);
        x = mvnrnd(l_mu, l_sig, N_REPLICAS);
        
        % Local Parameters
        steps_per_samp = round(local_para.COR_TIME / local_para.DT);
        n_warmup = ceil(local_para.WARM_TIME / local_para.COR_TIME);
        glob_step = 0;
        
        % --- Warm-up Loop ---
        noise_vals = [-local_para.NOISE, local_para.NOISE];
        
        % Variance Collection Variables
        sum_d = zeros(N_REPLICAS - 1, 1);
        sum_d2 = zeros(N_REPLICAS - 1, 1);
        cnt_d = zeros(N_REPLICAS - 1, 1);
        
        for ep = 1:n_warmup
            for s = 1:steps_per_samp
                % Generate random i_vec [i_1, i_2]
                i_vec = noise_vals(randi(2, 1, 2));
                
                glob_step = glob_step + 1;
                x = local_para.integrator(x, i_vec);
                if mod(glob_step, local_para.SWAP_INTERVAL_STEPS) == 0
                    [x, ~, att, d_vec] = perform_swap(x, local_para, rand() > 0.5, i_vec);
                    
                    % Accumulate Statistics for Variance Correction
                    mask = (att > 0.5);
                    sum_d(mask) = sum_d(mask) + d_vec(mask);
                    sum_d2(mask) = sum_d2(mask) + d_vec(mask).^2;
                    cnt_d(mask) = cnt_d(mask) + 1;
                end
            end
        end
        
        % Store State and Stats
        warmup_states{k} = x;
        warmup_stats{k} = struct('sum_d', sum_d, 'sum_d2', sum_d2, 'cnt_d', cnt_d);
    end
    
    % --- PHASE 2: AGGREGATE STATS & COMPUTE GLOBAL SIGMA ---
    total_sum_d  = zeros(N_REPLICAS - 1, 1);
    total_sum_d2 = zeros(N_REPLICAS - 1, 1);
    total_cnt_d  = zeros(N_REPLICAS - 1, 1);
    
    for k = 1:N_CORES
        stats = warmup_stats{k};
        total_sum_d  = total_sum_d  + stats.sum_d;
        total_sum_d2 = total_sum_d2 + stats.sum_d2;
        total_cnt_d  = total_cnt_d  + stats.cnt_d;
    end
    
    % Compute Global Sigma List (Standard Deviation of Delta H)
    mean_d = total_sum_d ./ max(1, total_cnt_d);
    var_d  = (total_sum_d2 ./ max(1, total_cnt_d)) - mean_d.^2;
    global_sigma_list = sqrt(max(0, var_d));
    
    fprintf('  [Sigma] Global Sigma List: %s\n', mat2str(global_sigma_list, 4));
    
    % --- PHASE 2: SAMPLING (CORRECT - Variance Corrected) ---
    fprintf('\n[PT-Parallel] Phase 2: Sampling with Global Variance Correction (Correct Mode)...\n');
    
    results_correct = cell(N_CORES, 1);
    
    parfor k = 1:N_CORES
        % Local copies
        local_para = para;
        local_samples_per_core = samples_per_core;
        
        % Retrieve State from Warm-up (Start from Equilibrated State)
        x = warmup_states{k};
        
        % Local Storage
        local_res = zeros(local_samples_per_core, N_REPLICAS, local_para.DIM);
        
        % Local Stats
        swap_attempts = zeros(N_REPLICAS - 1, 1);
        swap_accepts  = zeros(N_REPLICAS - 1, 1);
        
        % Local Parameters
        steps_per_samp = round(local_para.COR_TIME / local_para.DT);
        glob_step = 0; 
        
        core_tic = tic;
        
        % --- Sampling Loop ---
        noise_vals = [-local_para.NOISE, local_para.NOISE];
        for i = 1:local_samples_per_core
            for s = 1:steps_per_samp
                % Random i_vec
                i_vec = noise_vals(randi(2, 1, 2));
                
                glob_step = glob_step + 1;
                x = local_para.integrator(x, i_vec);
                
                if mod(glob_step, local_para.SWAP_INTERVAL_STEPS) == 0
                    swap_cnt = glob_step / local_para.SWAP_INTERVAL_STEPS;
                    is_even = mod(swap_cnt, 2) == 0;
                    % Pass GLOBAL sigma_list for Variance Correction
                    [x, acc, att] = perform_swap(x, local_para, is_even, i_vec, global_sigma_list);
                    
                    swap_attempts = swap_attempts + att;
                    swap_accepts  = swap_accepts + acc;
                end
            end
            
            % Store ALL replicas
            local_res(i, :, :) = x;
            
            % --- MONITORING (Core 1 Only) ---
            if k == 1
                if mod(i, 2000) == 0 || i == local_samples_per_core
                    safe_att = swap_attempts; safe_att(safe_att==0) = 1;
                    rates = swap_accepts ./ safe_att;
                    min_rate = min(rates);
                    
                    % We can't send struct easily in parfor without define var, simplified logging
                    fprintf('  [Correct] Iter %d/%d | Min Swap: %.1f%%\n', i, local_samples_per_core, min_rate*100);
                end
            end
        end
        
        results_correct{k} = local_res;
    end
    
    % --- PHASE 3: SAMPLING (NAIVE - No Variance Correction) ---
    fprintf('\n[PT-Parallel] Phase 3: Sampling WITHOUT Variance Correction (Naive Mode)...\n');
    
    results_naive = cell(N_CORES, 1);
    
    parfor k = 1:N_CORES
        % Local copies
        local_para = para;
        local_samples_per_core = samples_per_core;
        
        % Retrieve State from Warm-up (SAME STARTING POINT as Correct Phase)
        x = warmup_states{k};
        
        % Local Storage
        local_res = zeros(local_samples_per_core, N_REPLICAS, local_para.DIM);
        
        % Local Parameters
        steps_per_samp = round(local_para.COR_TIME / local_para.DT);
        glob_step = 0;
        
        % --- Sampling Loop ---
        noise_vals = [-local_para.NOISE, local_para.NOISE];
        % Use zero sigma for Naive
        naive_sigma_list = zeros(length(global_sigma_list), 1);
        
        for i = 1:local_samples_per_core
            for s = 1:steps_per_samp
                % Random i_vec
                i_vec = noise_vals(randi(2, 1, 2));
                
                glob_step = glob_step + 1;
                x = local_para.integrator(x, i_vec);
                
                if mod(glob_step, local_para.SWAP_INTERVAL_STEPS) == 0
                    swap_cnt = glob_step / local_para.SWAP_INTERVAL_STEPS;
                    is_even = mod(swap_cnt, 2) == 0;
                    % Pass ZERO sigma_list for Naive (No Correction)
                    [x, ~, ~] = perform_swap(x, local_para, is_even, i_vec, naive_sigma_list);
                end
            end
            
            % Store ALL replicas
            local_res(i, :, :) = x;
            
            % --- MONITORING (Core 1 Only) ---
            if k == 1
                if mod(i, 2000) == 0 || i == local_samples_per_core
                     fprintf('  [Naive]   Iter %d/%d\n', i, local_samples_per_core);
                end
            end
        end
        
        results_naive{k} = local_res;
    end
    
    total_time = toc(global_tic);
    fprintf('[PT-Parallel] Finished in %.1fs.\n', total_time);
    
    % =========================================================================
    % 3. Save Data (Sequential Files) for BOTH sets
    % =========================================================================
    
    % --- Helper Function to Save Results ---
    save_results(results_correct, 'correct', para, data_dir, N_REPLICAS);
    save_results(results_naive,   'naive',   para, data_dir, N_REPLICAS);
    
    % Prepare variable for Analysis (Default to correct)
    full_data = vertcat(results_correct{:}); 
    if size(full_data, 1) > para.MU_SIZE
        full_data = full_data(1:para.MU_SIZE, :, :);
    end
    samples_target = squeeze(full_data(:, end, :));
end

% ... (Analysis Section remains same) ...

% =========================================================================
% Local Functions
% =========================================================================

function save_results(results_cell, mode_str, para, data_dir, N_REPLICAS)
    fprintf('\n[Saving] Processing %s results...\n', mode_str);
    
    % Merge Data: [Total_Samples, Replicas, DIM]
    full_data = vertcat(results_cell{:}); 
    
    % Crop excessive samples if any
    if size(full_data, 1) > para.MU_SIZE
        full_data = full_data(1:para.MU_SIZE, :, :);
    end
    
    % Save Loop
    for k = 1:N_REPLICAS
        samples_k = squeeze(full_data(:, k, :));
        
        % Construct Variable Name and File Name
        file_idx = k - 1; 
        % Default legacy var name inside file, but file name distinguishes
        var_name = sprintf('samples_MU_%d', file_idx); 
        
        % File: {ABBR}_samples_{mode}_MU_{k}.mat
        file_name = sprintf('%s_samples_%s_MU_%d.mat', para.ABBR, mode_str, file_idx);
        file_path = fullfile(data_dir, file_name);
        
        S = struct();
        S.(var_name) = samples_k;
        save(file_path, '-struct', 'S');
        
        if k == 1 || k == N_REPLICAS || mod(k, 10) == 0
            fprintf('  -> Saved %s (%d samples)\n', file_name, size(samples_k, 1));
        end
    end
end

% =========================================================================
% 4. Analysis (Optional: Check Target Bias)
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

function [x, acc_vec, att_vec, delta_vec] = perform_swap(x, para, is_even_phase, i_vec, sigma_list)
    % UPDATED: Perform Swap for Linear Interpolation Potentials
    % U_lambda = (1-lambda)*U_base + lambda*U_target
    
    lambdas = para.BETA_LIST; % These are Lambdas now, not Betas
    M = length(lambdas);
    acc_vec = zeros(M - 1, 1);
    att_vec = zeros(M - 1, 1);
    delta_vec = zeros(M - 1, 1);
    
    if nargin < 5
        sigma_list = zeros(M-1, 1);
    end
    
    % We need U_target (GMM) and U_base (Gaussian) separately to compute delta correctly
    % U_base is cheap to compute on the fly.
    % U_target comes from para.compute_potential (which returns pure target)
    
    U_target_all = para.compute_potential(x, i_vec); % Vectorized [M, 1]
    
    % Compute U_base manually (Gaussian)
    % U = 0.5 * sum((x - mu)/sigma)^2
    z = (x - para.MEAN) ./ para.SIGMA;
    U_base_all = 0.5 * sum(z.^2, 2); 
    
    % Determine Pairs
    if is_even_phase
        pairs = 1:2:(M - 1);
    else
        pairs = 2:2:(M - 1);
    end
    
    % Variance Correction Parameter (from user request)
    tau = 0.5;
    
    for k = pairs
        idx_A = k;
        idx_B = k + 1;
        att_vec(k) = 1;
        
        lam_A = lambdas(idx_A);
        lam_B = lambdas(idx_B);
        
        % Energies at current positions
        % E(x, lambda) = (1-lam)*U0(x) + lam*U1(x)
        % Let D(x) = U1(x) - U0(x)
        % E(x, lambda) = U0(x) + lam * D(x)
        
        D_A = U_target_all(idx_A) - U_base_all(idx_A);
        D_B = U_target_all(idx_B) - U_base_all(idx_B);
        
        % Delta Energy for swapping A and B
        % Delta = E_new - E_old
        % E_old = E(xA, lamA) + E(xB, lamB)
        % E_new = E(xB, lamA) + E(xA, lamB)
        % Delta = [U0(xB) + lamA*D(xB) + U0(xA) + lamB*D(xA)] - [U0(xA) + lamA*D(xA) + U0(xB) + lamB*D(xB)]
        %       = lamA*(D(xB) - D(xA)) + lamB*(D(xA) - D(xB))
        %       = (lamA - lamB) * (D(xB) - D(xA))
        
        delta = (lam_A - lam_B) * (D_B - D_A);
        delta_vec(k) = delta;
        
        % Variance Correction
        % Effective Delta = Delta + (tau/2) * sigma^2
        if M > 1
             sigma_sq = sigma_list(k)^2;
        else
             sigma_sq = 0;
        end
        
        effective_delta = delta + (tau / 2) * sigma_sq;
        
        % Metropolis Criterion with Variance Correction
        if effective_delta <= 0 || rand() < exp(-effective_delta)
            % Swap Coordinates
            tmp_x = x(idx_A, :);
            x(idx_A, :) = x(idx_B, :);
            x(idx_B, :) = tmp_x;
            
            % Swap Cached Energies (Optimization)
            % Just swap the computed values in our temporary arrays to keep logic consistent 
            % (though we recompute next step anyway, this is just for logic correctness if we looped)
            tmp_Ut = U_target_all(idx_A); U_target_all(idx_A) = U_target_all(idx_B); U_target_all(idx_B) = tmp_Ut;
            tmp_Ub = U_base_all(idx_A);   U_base_all(idx_A)   = U_base_all(idx_B);   U_base_all(idx_B)   = tmp_Ub;
            
            acc_vec(k) = 1;
        end
    end
end