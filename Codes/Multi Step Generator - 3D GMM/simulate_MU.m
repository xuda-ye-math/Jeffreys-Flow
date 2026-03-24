clear; clc; rng(28);

% =========================================================================
% 0. Core Configuration
% =========================================================================
N_CORES = 5;       % Number of CPU cores
para = GMM_3D();   % Initialize Parameter Object (Updated for Linear Interp)

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
fprintf('[Check] Verifying existence of %d sample files...\n', N_REPLICAS);

for k = 0 : N_REPLICAS - 1
    % Construct expected filename: {ABBR}_samples_MU_{k}.mat
    fname = sprintf('%s_samples_MU_%d.mat', para.ABBR, k);
    fpath = fullfile(data_dir, fname);
    
    if ~exist(fpath, 'file')
        all_files_exist = false;
        fprintf('   -> Missing: %s\n', fname);
        break; % Stop checking if one is missing
    end
end

if all_files_exist
    fprintf('[Check] All %d datasets found. Skipping regeneration.\n', N_REPLICAS);
    
    % Load the final target file just to have variable 'samples_MU_M' available for analysis check
    target_idx = N_REPLICAS - 1;
    target_file = fullfile(data_dir, sprintf('%s_samples_MU_%d.mat', para.ABBR, target_idx));
    
    % Load dynamically
    loaded_struct = load(target_file);
    target_var_name = sprintf('samples_MU_%d', target_idx);
    
    % Assign to generic name for Analysis section
    if isfield(loaded_struct, target_var_name)
        samples_target = loaded_struct.(target_var_name);
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
    % Each cell will contain [samples_per_core, N_REPLICAS, DIM]
    results_all = cell(N_CORES, 1);
    
    % Start Global Timer
    global_tic = tic;
    
    parfor k = 1:N_CORES
        % Local copies
        local_para = para;
        local_samples_per_core = samples_per_core;
        
        % Local Storage: [Samples, Replicas, Dim]
        local_res = zeros(local_samples_per_core, N_REPLICAS, local_para.DIM);
        
        % Local Stats
        swap_attempts = zeros(N_REPLICAS - 1, 1);
        swap_accepts  = zeros(N_REPLICAS - 1, 1);
        
        % Initialize PT Ladder Locally
        l_mu = local_para.MEAN * ones(1, local_para.DIM);
        l_sig = local_para.SIGMA * eye(local_para.DIM);
        x = mvnrnd(l_mu, l_sig, N_REPLICAS);
        
        % Local Parameters
        steps_per_samp = round(local_para.COR_TIME / local_para.DT);
        n_warmup = ceil(local_para.WARM_TIME / local_para.COR_TIME);
        glob_step = 0;
        
        % Independent Timer for this Core
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
            
            % Store ALL replicas
            local_res(i, :, :) = x;
            
            % --- MONITORING (Core 1 Only) ---
            if k == 1
                if mod(i, 2000) == 0 || i == local_samples_per_core
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
    % 3. Save Data (Sequential Files)
    % =========================================================================
    fprintf('\n[Saving] Processing and saving %d datasets...\n', N_REPLICAS);
    
    % Merge Data: [Total_Samples, Replicas, DIM]
    % vertcat works on the first dimension
    full_data = vertcat(results_all{:}); 
    
    % Crop excessive samples if any
    if size(full_data, 1) > para.MU_SIZE
        full_data = full_data(1:para.MU_SIZE, :, :);
    end
    
    % Save Loop
    for k = 1:N_REPLICAS
        % Extract samples for replica k
        samples_k = squeeze(full_data(:, k, :));
        
        % Construct Variable Name and File Name
        % Index is 0-based for files (MU_0 ... MU_M)
        file_idx = k - 1; 
        var_name = sprintf('samples_MU_%d', file_idx);
        file_name = sprintf('%s_%s.mat', para.ABBR, var_name);
        file_path = fullfile(data_dir, file_name);
        
        % Save
        S = struct();
        S.(var_name) = samples_k;
        save(file_path, '-struct', 'S');
        
        if k == 1 || k == N_REPLICAS || mod(k, 10) == 0
            fprintf('  -> Saved %s (%d samples)\n', file_name, size(samples_k, 1));
        end
    end
    
    % Prepare variable for Analysis
    samples_target = squeeze(full_data(:, end, :));
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

function [x, acc_vec, att_vec] = perform_swap(x, para, is_even_phase)
    % UPDATED: Perform Swap for Linear Interpolation Potentials
    % U_lambda = (1-lambda)*U_base + lambda*U_target
    
    lambdas = para.BETA_LIST; % These are Lambdas now, not Betas
    M = length(lambdas);
    acc_vec = zeros(M - 1, 1);
    att_vec = zeros(M - 1, 1);
    
    % We need U_target (GMM) and U_base (Gaussian) separately to compute delta correctly
    % U_base is cheap to compute on the fly.
    % U_target comes from para.compute_potential (which returns pure target)
    
    U_target_all = para.compute_potential(x); % Vectorized [M, 1]
    
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
        
        % Metropolis Criterion
        if delta <= 0 || rand() < exp(-delta)
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