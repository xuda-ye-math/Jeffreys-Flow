clear; clc; rng(27);

% =========================================================================
% 1. Initialization
% =========================================================================

% Initialize Problem Object
% Parameters are encapsulated within the class instance
para = Periodic_Well();
fprintf('Initializing PT Simulation for %s...\n', para.NAME);
para.disp_info();

% Directory Setup
data_dir = './data';
if ~exist(data_dir, 'dir'); mkdir(data_dir); end

save_path_0 = fullfile(data_dir, sprintf('%s_samples_MU_0.mat', para.ABBR));
save_path_1 = fullfile(data_dir, sprintf('%s_samples_MU_1.mat', para.ABBR));

% =========================================================================
% 2. Main Simulation Logic
% =========================================================================

if exist(save_path_0, 'file') && exist(save_path_1, 'file')
    % ---- Case A: Load Existing Data ----
    fprintf('\n[Check] Found existing data. Loading...\n');
    load(save_path_1, 'samples_MU_1');
    
else
    % ---- Case B: Generate New Data ----
    
    % --- Part I: Base Distribution (MU_0) ---
    fprintf('\n[Sampling MU_0] Generating Uniform Base Samples...\n');
    
    % Retrieve plotting limits (Domain Boundaries)
    % For Periodic Well, this is [-pi, pi]
    x_min = para.X_LIM_PLOT(1);
    x_max = para.X_LIM_PLOT(2);
    domain_width = x_max - x_min;
    
    % Uniform Sampling: min + (max-min) * rand
    samples_MU_0 = x_min + domain_width * rand(para.MU_SIZE, para.DIM);
    
    save(save_path_0, 'samples_MU_0');
    fprintf('[Result] Saved samples to %s\n', save_path_0);
    
    % --- Part II: Parallel Tempering (MU_1) ---
    
    % Initialization
    % Initialize PT particles uniformly in the periodic domain
    x = x_min + domain_width * rand(para.PT_M, para.DIM); 
    
    samples_MU_1 = zeros(para.MU_SIZE, para.DIM);
    
    % Statistics Tracking
    swap_stats.attempts = zeros(para.PT_M - 1, 1);
    swap_stats.accepts  = zeros(para.PT_M - 1, 1);
    
    % Timing
    steps_per_sample = round(para.COR_TIME / para.DT);
    num_warmup_epochs = ceil(para.WARM_TIME / para.COR_TIME);
    global_step = 0;
    
    % --- Warm-up Phase ---
    fprintf('\n[PT] Warming up (Physical Time: %.1f)...\n', para.WARM_TIME);
    
    for epoch = 1:num_warmup_epochs
        for s = 1:steps_per_sample
            global_step = global_step + 1;
            
            % 1. Evolution (Uses internal parameters)
            x = para.integrator(x);
            
            % 2. Swap (Random Phase for warmup)
            if mod(global_step, para.SWAP_INTERVAL_STEPS) == 0
                is_even = rand() > 0.5;
                x = perform_swap(x, para, is_even);
            end
        end
    end
    
    % --- Main Sampling Phase ---
    fprintf('[PT] Starting Main Sampling Loop (%d samples)...\n', para.MU_SIZE);
    tic;
    
    for i = 1:para.MU_SIZE
        for s = 1:steps_per_sample
            global_step = global_step + 1;
            
            % 1. Evolution
            x = para.integrator(x);
            
            % 2. Swap (Deterministic Alternating Phase)
            if mod(global_step, para.SWAP_INTERVAL_STEPS) == 0
                swap_count = global_step / para.SWAP_INTERVAL_STEPS;
                is_even = mod(swap_count, 2) == 0;
                
                [x, acc, att] = perform_swap(x, para, is_even);
                
                swap_stats.attempts = swap_stats.attempts + att;
                swap_stats.accepts  = swap_stats.accepts + acc;
            end
        end
        
        % 3. Collect Sample (Lowest Temperature / Last Replica)
        samples_MU_1(i, :) = x(end, :);
        
        % 4. Logging
        if mod(i, 1000) == 0 || i == para.MU_SIZE
            elapsed = toc;
            safe_att = swap_stats.attempts; safe_att(safe_att==0) = 1;
            rates = swap_stats.accepts ./ safe_att;
            
            fprintf('  Progress: %5d / %d | Min Swap: %.1f%% | Time: %.1fs\n', ...
                i, para.MU_SIZE, min(rates)*100, elapsed);
        end
    end
    
    % --- Post-Processing: Periodic Wrapping ---
    % Ensure all saved samples are strictly within [-pi, pi]
    % This handles any potential drift during the simulation steps
    samples_MU_1 = mod(samples_MU_1 + pi, 2*pi) - pi;
    
    save(save_path_1, 'samples_MU_1');
    fprintf('\n[Result] Saved samples to %s\n', save_path_1);
    
end

% Compute Bias (if function exists)
if exist('compute_bias', 'file')
    compute_bias(para, samples_MU_1, 1.0);
end

% =========================================================================
% Helper Functions
% =========================================================================

function [x, acc_vec, att_vec] = perform_swap(x, para, is_even_phase)
    % Perform Metropolis-Hastings swap between adjacent replicas
    
    M = para.PT_M;
    acc_vec = zeros(M - 1, 1);
    att_vec = zeros(M - 1, 1);
    
    % Compute Energies
    U = para.compute_potential(x);
    betas = para.BETA_LIST; % Retrieve betas from object
    
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
        
        % Metropolis Criterion
        delta = (betas(idx_A) - betas(idx_B)) * (U(idx_A) - U(idx_B));
        
        if delta > 0 || rand() < exp(delta)
            % Accept Swap
            tmp_x = x(idx_A, :);
            x(idx_A, :) = x(idx_B, :);
            x(idx_B, :) = tmp_x;
            
            % Optimization: Update local energy to avoid recompute? 
            % (Keeping simple here as per request to keep functionality)
            temp_u = U(idx_A);
            U(idx_A) = U(idx_B);
            U(idx_B) = temp_u;
            
            acc_vec(k) = 1;
        end
    end
end