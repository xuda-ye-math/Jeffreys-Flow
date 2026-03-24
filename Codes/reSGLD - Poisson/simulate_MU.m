% simulate_MU.m
% 
% Parallel Tempering (reSGLD) for 2D Screened Poisson Source Inversion.
% 1 Core = 1 Temperature Architecture.
% Uses matfile chunk streaming to reduce memory footprint.
% Includes an initial burn-in phase.

clear; clc; close all;
rng(42); % Fixed seed for reproducibility

% =========================================================================
% 0. Configuration & Initialization
% =========================================================================
fprintf('--- Initializing Naive reSGLD Sampler (Chunk Streaming) ---\n');

% Load Parameters and Data
data_dir = './data';
para_file = fullfile(data_dir, 'para.mat');

if ~exist(para_file, 'file')
    error('Configuration file %s not found. Run parameters.m first.', para_file);
end

data = load(para_file);
para = data.para;

% Force delete all existing sample data from previous runs to avoid accidental appends or dimension mismatches.
old_files = dir(fullfile(data_dir, 'samples_MU_*.mat'));
if ~isempty(old_files)
    fprintf('Purging %d old sample files from disk...\n', length(old_files));
    for i = 1:length(old_files)
        delete(fullfile(data_dir, old_files(i).name));
    end
end
fprintf('Working directory clean. Ready for new simulation.\n');

num_chains = para.M + 1;
beta_list = para.beta_list;

fprintf('Detected %d Total Chains from parameters (beta_0 to beta_%d).\n', num_chains, para.M);

% Dimensions and loops
dim = 8;
MU_SIZE = para.total_samples;
swap_interval = para.swap_interval_steps;
record_interval = para.record_interval;

% Stream to disk every N records to save memory
flush_interval_records = 1000;
chunk_buffer = zeros(num_chains, flush_interval_records, dim);
buffer_idx = 0; % How many records currently in the buffer

total_sgld_steps = MU_SIZE * record_interval;
num_swap_cycles = ceil(total_sgld_steps / swap_interval);

% =========================================================================
% 1. Memory & Disk Allocation
% =========================================================================
fprintf('Preparing matfiles for streaming...\n');
mat_objs = cell(num_chains, 1);
for m = 1:num_chains
    file_idx = m - 1;
    file_name = sprintf('samples_MU_%d.mat', file_idx);
    file_path = fullfile(data_dir, file_name);
    
    var_name = sprintf('samples_MU_%d', file_idx);
    
    % Initialize the matfile with a 1x1 dummy to establish the variable,
    % then we will append to it. MAT-files must be v7.3 to support append.
    % Actually, we can preallocate the exact size directly using matfile.
    
    % Delete it if it already exists to start fresh
    if exist(file_path, 'file')
        delete(file_path);
    end
    
    mobj = matfile(file_path, 'Writable', true);
    % Preallocate the full array with zeros to optimize disk sector layout
    mobj.(var_name) = zeros(MU_SIZE, dim); 
    mat_objs{m} = mobj;
end

% Initialize random starting positions for all chains within [0, 1]^8
current_X = rand(num_chains, dim);

recorded_counts = zeros(num_chains, 1);
swap_attempts = zeros(num_chains - 1, 1);
swap_accepts  = zeros(num_chains - 1, 1);

% =========================================================================
% 2. Setup Parallel Environment
% =========================================================================
p = gcp('nocreate');
if isempty(p) || p.NumWorkers < num_chains
    delete(gcp('nocreate'));
    c = parcluster('local');
    if c.NumWorkers < num_chains
        c.NumWorkers = num_chains;
    end
    try
        parpool(c, num_chains);
    catch ME
        warning('Failed to allocate %d workers: %s. Reverting to default.', num_chains, ME.message);
        parpool;
    end
end
N_CORES = gcp().NumWorkers;
fprintf('Parallel pool running with %d workers.\n\n', N_CORES);

% =========================================================================
% 3. Burn-in Phase
% =========================================================================
burn_in_steps = 500;
fprintf('--- Starting Burn-in Phase (%d steps) ---\n', burn_in_steps);

burn_in_cycles = ceil(burn_in_steps / swap_interval);

for cycle = 1:burn_in_cycles
    chunk_end_X = zeros(num_chains, dim);
    
    parfor m = 1:num_chains
        beta_m = beta_list(m);
        theta_m = current_X(m, :)';
        dt = para.dt;
        sigma_noise = para.sigma_noise;
        batch_size = para.batch_size;
        
        local_solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);
        
        for step = 1:swap_interval 
            if beta_m == 0
                theta_m = rand(8, 1);
            else
                batch_indices = randperm(para.total_obs, batch_size);
                grad_U = get_gradient_mixed_partial(theta_m, local_solver, para.y_obs, para.obs_indices, ...
                                                    sigma_noise, batch_indices, beta_m);
                eta = randn(8, 1);
                theta_m = theta_m - dt * grad_U + sqrt(2 * dt) * eta;
                
                theta_m = abs(theta_m);
                theta_m = 1 - abs(1 - theta_m);
            end
        end
        chunk_end_X(m, :) = theta_m';
    end
    
    current_X = chunk_end_X;
    
    % Perform Swap (Same as main loop, but we don't track stats deeply here)
    full_batch = 1:para.total_obs;
    U_targets = zeros(num_chains, 1);
    master_solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);
    
    for m = 1:num_chains
        U_targets(m) = get_potential_mixed_partial(current_X(m, :)', master_solver, ...
            para.y_obs, para.obs_indices, para.sigma_noise, full_batch, 1.0);
    end
    
    is_even = mod(cycle, 2) == 0;
    if is_even
        pairs = 1:2:(num_chains - 1);
    else
        pairs = 2:2:(num_chains - 1);
    end
    
    for k = pairs
        idx_A = k; idx_B = k + 1;
        beta_A = beta_list(idx_A); beta_B = beta_list(idx_B);
        U_A = U_targets(idx_A); U_B = U_targets(idx_B);
        delta = (beta_A - beta_B) * (U_B - U_A);
        
        if delta <= 0 || rand() < exp(-delta)
            temp_X = current_X(idx_A, :);
            current_X(idx_A, :) = current_X(idx_B, :);
            current_X(idx_B, :) = temp_X;
        end
    end
    
    if mod(cycle, 10) == 0 || cycle == burn_in_cycles
        fprintf('  [Burn-in %4d / %4d cycles completed]\n', cycle, burn_in_cycles);
    end
end

fprintf('Burn-in complete. Resetting stats and starting formal recording...\n\n');

% =========================================================================
% 4. Main Parallel Tempering Loop (Formal Recording)
% =========================================================================
global_step = 0;
global_tic = tic;

for cycle = 1:num_swap_cycles
    
    start_step = global_step + 1;
    end_step = global_step + swap_interval;
    
    chunk_steps = start_step:end_step;
    record_flags = mod(chunk_steps, record_interval) == 0;
    num_records_this_chunk = sum(record_flags);
    
    chunk_end_X = zeros(num_chains, dim);
    chunk_recorded_X = zeros(num_chains, num_records_this_chunk, dim);
    
    parfor m = 1:num_chains
        beta_m = beta_list(m);
        theta_m = current_X(m, :)'; 
        dt = para.dt;
        sigma_noise = para.sigma_noise;
        batch_size = para.batch_size;
        
        local_solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);
        
        local_records = zeros(num_records_this_chunk, dim);
        record_idx = 0;
        
        for step = 1:swap_interval 
            curr_global_step = start_step + step - 1;
            
            if beta_m == 0
                theta_m = rand(8, 1);
            else
                batch_indices = randperm(para.total_obs, batch_size);
                grad_U = get_gradient_mixed_partial(theta_m, local_solver, para.y_obs, para.obs_indices, ...
                                                    sigma_noise, batch_indices, beta_m);
                eta = randn(8, 1);
                theta_m = theta_m - dt * grad_U + sqrt(2 * dt) * eta;
                
                theta_m = abs(theta_m);
                theta_m = 1 - abs(1 - theta_m);
            end
            
            if mod(curr_global_step, record_interval) == 0
                record_idx = record_idx + 1;
                local_records(record_idx, :) = theta_m';
            end
        end
        chunk_end_X(m, :) = theta_m';
        chunk_recorded_X(m, :, :) = local_records;
    end
    
    % Buffer the records
    if num_records_this_chunk > 0
        end_buffer_idx = buffer_idx + num_records_this_chunk;
        
        % Ensure we don't exceed the chunk buffer allocating size
        if end_buffer_idx <= flush_interval_records
             chunk_buffer(:, buffer_idx+1 : end_buffer_idx, :) = chunk_recorded_X;
             buffer_idx = end_buffer_idx;
        else
             % Edge case if it somehow overshoots (unlikely if neatly divisible)
             space_left = flush_interval_records - buffer_idx;
             chunk_buffer(:, buffer_idx+1 : end, :) = chunk_recorded_X(:, 1:space_left, :);
             buffer_idx = flush_interval_records; % Triggers flush below
        end
    end
    
    current_X = chunk_end_X;
    global_step = end_step;
    
    % -------------------------------------------------------------
    % Flush Buffer to Disk (matfile)
    % -------------------------------------------------------------
    % Every time the buffer reaches flush_interval_records (e.g., 1000)
    % Or if this is the very last cycle!
    is_final_cycle = (cycle == num_swap_cycles);
    
    if buffer_idx >= flush_interval_records || (is_final_cycle && buffer_idx > 0)
        
        for m = 1:num_chains
            var_name = sprintf('samples_MU_%d', m - 1);
            
            start_idx = recorded_counts(m) + 1;
            end_idx = recorded_counts(m) + buffer_idx;
            
            % Write exactly this block to the disk file
            mat_objs{m}.(var_name)(start_idx:end_idx, :) = squeeze(chunk_buffer(m, 1:buffer_idx, :));
            
            recorded_counts(m) = end_idx;
        end
        
        % Reset buffer
        buffer_idx = 0;
    end
    
    % -------------------------------------------------------------
    % Replica Exchange (Serial)
    % -------------------------------------------------------------
    full_batch = 1:para.total_obs;
    U_targets = zeros(num_chains, 1);
    master_solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);
    
    for m = 1:num_chains
        U_targets(m) = get_potential_mixed_partial(current_X(m, :)', master_solver, ...
            para.y_obs, para.obs_indices, para.sigma_noise, full_batch, 1.0);
    end
    
    is_even = mod(cycle, 2) == 0;
    if is_even
        pairs = 1:2:(num_chains - 1);
    else
        pairs = 2:2:(num_chains - 1);
    end
    
    for k = pairs
        idx_A = k; idx_B = k + 1;
        beta_A = beta_list(idx_A); beta_B = beta_list(idx_B);
        U_A = U_targets(idx_A); U_B = U_targets(idx_B);
        
        delta = (beta_A - beta_B) * (U_B - U_A);
        swap_attempts(k) = swap_attempts(k) + 1;
        
        if delta <= 0 || rand() < exp(-delta)
            temp_X = current_X(idx_A, :);
            current_X(idx_A, :) = current_X(idx_B, :);
            current_X(idx_B, :) = temp_X;
            swap_accepts(k) = swap_accepts(k) + 1;
        end
    end
    
    % -------------------------------------------------------------
    % Logging
    % -------------------------------------------------------------
    total_recorded_now = recorded_counts(1) + buffer_idx;
    if (num_records_this_chunk > 0 && floor(total_recorded_now / 10) > floor((total_recorded_now - num_records_this_chunk) / 10)) || cycle == num_swap_cycles
        safe_att = swap_attempts;
        safe_att(safe_att == 0) = 1;
        rates = swap_accepts ./ safe_att;
        min_rate = min(rates) * 100;
        
        elapsed = toc(global_tic);
        progress = max(1, total_recorded_now) / MU_SIZE;
        eta_seconds = max(0, (elapsed / progress) - elapsed);
        eta_hours = floor(eta_seconds / 3600);
        eta_mins = floor(mod(eta_seconds, 3600) / 60);
        
        fprintf('  [Recorded %6d / %6d] Global Step: %8d | Min Swap: %5.1f%% | Time: %6.1fs | ETA: %02dh%02dm\n', ...
            total_recorded_now, MU_SIZE, global_step, min_rate, elapsed, eta_hours, eta_mins);
    end
end

fprintf('\nAll Data Processed and Saved to Disk Successfully.\n');
