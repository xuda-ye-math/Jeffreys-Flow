clear; clc;

para_path = './data/para.mat';
if ~exist(para_path, 'file')
    error('Configuration para.mat not found.');
end
load(para_path); % Loads 'para' struct

M = para.M;
data_dir = './data';

% Define methods to report
methods = {'e', 'f', 's'};
method_names = {'Exact (NU-e)', 'Full Pot. (NU-f)', 'Stochastic (NU-s)'};
num_methods = length(methods);

% Initialize CESS matrix: rows=methods, cols=steps(1 to M)
cess_matrix = NaN(num_methods, M);

% Load CESS values
for i = 1:num_methods
    mode_str = methods{i};
    for k = 1:M
        wname = sprintf('weights_%s_NU_%d.mat', mode_str, k);
        fpath = fullfile(data_dir, wname);
        if exist(fpath, 'file')
            try
                tmp = load(fpath, 'cess');
                if isfield(tmp, 'cess')
                    cess_matrix(i, k) = tmp.cess;
                end
            catch
                % Do nothing, remains NaN
            end
        end
    end
end

% =========================================================================
% Print Formatted Table
% =========================================================================
fprintf('\n=========================================================================================================\n');
fprintf('  Conditional Effective Sample Size (CESS %%) Across Stages (k = 1 ... %d)\n', M);
fprintf('=========================================================================================================\n');

% Print Header
fprintf('%-25s', 'Method \ Step (k)');
for k = 1:M
    fprintf(' | %6s %d', 'Step', k);
end
fprintf('\n-------------------------');
for k = 1:M
    fprintf('---------');
end
fprintf('\n');

% Print Data Rows
for i = 1:num_methods
    fprintf('%-25s', method_names{i});
    for k = 1:M
        val = cess_matrix(i, k);
        if isnan(val)
            fprintf(' | %8s', 'N/A');
        else
            fprintf(' | %7.2f%%', val * 100);
        end
    end
    fprintf('\n');
end

fprintf('=========================================================================================================\n\n');
