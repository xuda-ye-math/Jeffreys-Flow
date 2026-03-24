
% print_CESS.m:
% Prints three rows:
% 1st Row: Beta values (from Gaussian to Target, BETA_LIST)
% 2nd Row: Corresponding CESS (calculated from F_1 onwards)
% 3rd Row: Current distribution's ESS.
% The output format should resemble a table.

clear; clc;

% Setup Parameters
data_dir = './data';
abbr = 'NR'; 
beta_file_name = sprintf('%s_BETA_LIST.mat', abbr);
beta_file_path = fullfile(data_dir, beta_file_name);

% Load Alpha (Beta) List
if ~exist(beta_file_path, 'file')
    error('File %s not found. Please run the simulation first.', beta_file_name);
end
loaded_struct = load(beta_file_path);
if isfield(loaded_struct, 'BETA_LIST')
    BETA_LIST = loaded_struct.BETA_LIST;
else
    error('BETA_LIST variable not found in %s', beta_file_name);
end

M = length(BETA_LIST) - 1;
fprintf('Start Printing CESS/ESS for M = %d steps...\n\n', M);

% Initialize arrays for storing data
beta_vals = zeros(1, M+1);
cess_vals = zeros(1, M+1);
ess_vals  = zeros(1, M+1);

% Iterate through each step to read data
for k = 0 : M
    % Construct filename: NR_weights_NU_k.mat
    w_filename = sprintf('%s_weights_NU_%d.mat', abbr, k);
    w_path = fullfile(data_dir, w_filename);
    
    if ~exist(w_path, 'file')
        fprintf('Warning: File %s not found. Filling with NaN.\n', w_filename);
        cess_vals(k+1) = NaN;
        ess_vals(k+1)  = NaN;
    else
        % Load .mat file
        tmp = load(w_path);
        
        % Read Beta
        beta_vals(k+1) = BETA_LIST(k+1);
        
        % Read ESS / CESS
        % Note: train_flow.py saves 'cess' and 'ess' as scalars inside the .mat
        if isfield(tmp, 'cess')
            cess_vals(k+1) = tmp.cess;
        else
            cess_vals(k+1) = NaN;
        end
        
        if isfield(tmp, 'ess')
            ess_vals(k+1) = tmp.ess;
        else
            ess_vals(k+1) = NaN;
        end
    end
end

% -------------------------------------------------------------------------
% Print Table
% -------------------------------------------------------------------------

% Print Header (Indices)
fprintf('%-10s', 'Step:');
for k = 0 : M
    fprintf('%8d', k);
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 10 + 8*(M+1)));

% Row 1: Beta
fprintf('%-10s', 'Beta:');
for k = 0 : M
    fprintf('%8.3f', beta_vals(k+1));
end
fprintf('\n');

% Row 2: CESS
% "Starting from F_1" -> implies k=0 is N/A or '-'
fprintf('%-10s', 'CESS:');
% k=0 (Base) usually doesn't have a transition CESS
fprintf('%8s', '-'); 
for k = 1 : M
    fprintf('%8.1f%%', cess_vals(k+1) * 100);
end
fprintf('\n');

% Row 3: ESS
fprintf('%-10s', 'ESS:');
for k = 0 : M
    fprintf('%8.1f%%', ess_vals(k+1) * 100);
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 10 + 8*(M+1)));
