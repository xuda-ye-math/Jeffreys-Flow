clear; clc;

% =========================================================================
% 1. Configuration
% =========================================================================
addpath(genpath('./potential'));

para = GMM_2D();
abbr = para.ABBR;
data_dir = './data';

% Load Beta List
beta_file = fullfile(data_dir, sprintf('%s_BETA_LIST.mat', abbr));
if ~exist(beta_file, 'file')
    error('File %s not found.', beta_file);
end
tmp = load(beta_file);
BETA_LIST = tmp.BETA_LIST;
M = length(BETA_LIST) - 1;

fprintf('Printing Metrics for M=%d steps...\n', M);

% =========================================================================
% 2. Iteration & Metric Collection
% =========================================================================

% Modes: 
% 1. reSGLD (MU) - Naive
% 2. reSGLD (MU) - Correct
% 3. JF No-MALA (NU) - Naive
% 4. JF No-MALA (NU) - Correct
% 5. JF MALA (NU) - Naive
% 6. JF MALA (NU) - Correct

% We will store bias arrays for all steps first
bias_resgld_n = zeros(1, M+1);
bias_resgld_c = zeros(1, M+1);
bias_jf_nm_n  = zeros(1, M+1);
bias_jf_nm_c  = zeros(1, M+1);
bias_jf_m_n   = zeros(1, M+1);
bias_jf_m_c   = zeros(1, M+1);

ess_jf_nm_n = zeros(1, M+1);
ess_jf_nm_c = zeros(1, M+1);
ess_jf_m_n  = zeros(1, M+1);
ess_jf_m_c  = zeros(1, M+1);

for k = 0 : M
    beta_val = BETA_LIST(k+1);
    
    % --- 1. reSGLD Naive ---
    bias_resgld_n(k+1) = get_bias(para, abbr, 'naive', 'MU', k, beta_val);
    
    % --- 2. reSGLD Correct ---
    bias_resgld_c(k+1) = get_bias(para, abbr, 'correct', 'MU', k, beta_val);
    
    % --- 3. JF No-MALA Naive ---
    [bias_jf_nm_n(k+1), ess_jf_nm_n(k+1), ~] = get_metrics(para, abbr, 'naive_no_mala', 'NU', k, beta_val);
    
    % --- 4. JF No-MALA Correct ---
    [bias_jf_nm_c(k+1), ess_jf_nm_c(k+1), ~] = get_metrics(para, abbr, 'correct_no_mala', 'NU', k, beta_val);
    
    % --- 5. JF MALA Naive ---
    [bias_jf_m_n(k+1), ess_jf_m_n(k+1), ~] = get_metrics(para, abbr, 'naive_mala', 'NU', k, beta_val);
    
    % --- 6. JF MALA Correct ---
    [bias_jf_m_c(k+1), ess_jf_m_c(k+1), ~] = get_metrics(para, abbr, 'correct_mala', 'NU', k, beta_val);
end

% =========================================================================
% 3. Print Results (Tab Separated for easy Excel Paste)
% =========================================================================
fprintf('\n');

% Print Step
fprintf('%s\t%s\t', 'Method', 'Step');
for k = 0 : M
    fprintf('%d\t', k);
end
fprintf('\n');

% Print Beta
fprintf('%s\t%s\t', '', 'beta');
for k = 0 : M
    fprintf('%.4g\t', BETA_LIST(k+1));
end
fprintf('\n');

% Print reSGLD
fprintf('%s\t%s\t', 'reSGLD', 'naive');
fprintf('%.4e\t', bias_resgld_n);
fprintf('\n');

fprintf('%s\t%s\t', '', 'correct');
fprintf('%.4e\t', bias_resgld_c);
fprintf('\n');

% Print JF No MALA
fprintf('%s\t%s\t', 'JF (No MALA)', 'naive');
fprintf('%.4e\t', bias_jf_nm_n);
fprintf('\n');

fprintf('%s\t%s\t', '', 'correct');
fprintf('%.4e\t', bias_jf_nm_c);
fprintf('\n');

% Print JF MALA
fprintf('%s\t%s\t', 'JF (MALA)', 'naive');
fprintf('%.4e\t', bias_jf_m_n);
fprintf('\n');

fprintf('%s\t%s\t', '', 'correct');
fprintf('%.4e\t', bias_jf_m_c);
fprintf('\n');
fprintf('\n');

% =========================================================================
% 4. Print ESS Results (Tab Separated)
% =========================================================================
fprintf('\n');

% Print Step
fprintf('%s\t%s\t', 'Method', 'Step');
for k = 0 : M
    fprintf('%d\t', k);
end
fprintf('\n');

% Print Beta
fprintf('%s\t%s\t', '', 'beta');
for k = 0 : M
    fprintf('%.4g\t', BETA_LIST(k+1));
end
fprintf('\n');

% Print JF No MALA
fprintf('%s\t%s\t', 'JF (No MALA)', 'naive');
fprintf('%.4g\t', ess_jf_nm_n);
fprintf('\n');

fprintf('%s\t%s\t', '', 'correct');
fprintf('%.4g\t', ess_jf_nm_c);
fprintf('\n');

% Print JF MALA
fprintf('%s\t%s\t', 'JF (MALA)', 'naive');
fprintf('%.4g\t', ess_jf_m_n);
fprintf('\n');

fprintf('%s\t%s\t', '', 'correct');
fprintf('%.4g\t', ess_jf_m_c);
fprintf('\n');
fprintf('\n');


% =========================================================================
% Helper Functions
% =========================================================================

function bias = get_bias(para, abbr, mode_str, type, k, beta)
    % Compute bias only (for MU or unweighted)
    fname = sprintf('%s_samples_%s_%s_%d.mat', abbr, mode_str, type, k);
    fpath = fullfile('./data', fname);
    
    bias = NaN;
    if exist(fpath, 'file')
        tmp = load(fpath);
        if isfield(tmp, 'samples'), s=tmp.samples; else, vars=fieldnames(tmp); s=tmp.(vars{1}); end
        if ~isempty(s)
            bias = compute_bias(para, s, beta, []);
        end
    end
end

function [bias, ess, cess] = get_metrics(para, abbr, mode_str, type, k, beta)
    % Compute bias (weighted), ESS, CESS
    fname_s = sprintf('%s_samples_%s_%s_%d.mat', abbr, mode_str, type, k);
    fname_w = sprintf('%s_weights_%s_%s_%d.mat', abbr, mode_str, type, k);
    
    path_s = fullfile('./data', fname_s);
    path_w = fullfile('./data', fname_w);
    
    bias = NaN; ess = NaN; cess = NaN;
    
    if exist(path_s, 'file') && exist(path_w, 'file')
        % Samples
        tmp_s = load(path_s);
        if isfield(tmp_s, 'samples'), s=tmp_s.samples; else, vars=fieldnames(tmp_s); s=tmp_s.(vars{1}); end
        
        % Weights
        tmp_w = load(path_w);
        if isfield(tmp_w, 'weights'), log_w=tmp_w.weights; else, vars=fieldnames(tmp_w); log_w=tmp_w.(vars{1}); end
        
        % Metrics from file (ESS/CESS saved in struct)
        if isfield(tmp_w, 'ess'), ess = tmp_w.ess; end
        if isfield(tmp_w, 'cess'), cess = tmp_w.cess; end
        
        % Bias (Compute Weighted)
        if ~isempty(s)
            % Normalize weights
             log_w = double(log_w(:));
             w = exp(log_w - max(log_w));
             w_norm = w / sum(w);
             
             bias = compute_bias(para, s, beta, w_norm);
        end
    end
end
