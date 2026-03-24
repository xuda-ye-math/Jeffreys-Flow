function l2_bias = compute_bias(para, samples, beta, weights)
    % COMPUTE_BIAS Computes the L2 bias of samples against the ground truth.
    %
    % Description:
    %   Calculates the L2 norm of the error vector between sample estimates
    %   and the ground truth expectations for a set of test functions.
    %   The ground truth is obtained via high-resolution numerical integration
    %   on the potential energy landscape defined in 'para'.
    %
    %   UPDATED: Uses 8 periodic test functions (Fourier basis up to order 2).
    %
    % Usage:
    %   1. l2_bias = compute_bias(para, samples, beta)
    %      -> For unweighted samples (e.g., standard MCMC).
    %   2. l2_bias = compute_bias(para, samples, beta, weights)
    %      -> For weighted samples (e.g., Importance Sampling).
    %
    % Inputs:
    %   para    : Potential object (e.g., Periodic_Well). Must contain:
    %             - Properties: X_LIM_PLOT, Y_LIM_PLOT
    %             - Method: compute_potential(x)
    %   samples : [N, DIM] Matrix of sample points.
    %   beta    : Scalar inverse temperature.
    %   weights : (Optional) [N, 1] Vector of importance weights.
    %
    % Output:
    %   l2_bias : Scalar representing the Euclidean norm (L2) of the bias.

    % =========================================================================
    % 1. Initialization & Argument Parsing
    % =========================================================================
    
    % Grid resolution for numerical integration (Ground Truth)
    % Note: 5000x5000 points provides high precision for 2D integration
    GRID_SIZE = 5000; 

    % Handle optional weights argument
    if nargin < 4
        weights = [];
    end
    
    if ~isempty(weights)
        weights = weights(:); % Ensure column vector
    end

    % =========================================================================
    % 2. Compute Ground Truth (Numerical Integration)
    % =========================================================================
    
    % Retrieve integration boundaries from the potential object
    % Using PLOT limits as they define the periodic domain [-pi, pi]
    x_lim = para.X_LIM_PLOT;
    y_lim = para.Y_LIM_PLOT;
    
    % Generate Mesh Grid
    vals_x = linspace(x_lim(1), x_lim(2), GRID_SIZE);
    vals_y = linspace(y_lim(1), y_lim(2), GRID_SIZE);
    [X, Y] = ndgrid(vals_x, vals_y);
    
    % Flatten grid for vectorized evaluation: [TotalPoints, 2]
    grid_points = [X(:), Y(:)];
    
    % Evaluate Potential Energy
    U = para.compute_potential(grid_points);
    
    % Compute Boltzmann Weights: w = exp(-beta * U)
    % Optimization: Use Log-Sum-Exp trick for numerical stability
    log_weights = -beta * U;
    max_log = max(log_weights); 
    w_grid = exp(log_weights - max_log);
    
    % Partition function (normalization constant for the grid)
    Z = sum(w_grid);

    % =========================================================================
    % 3. Define Test Functions (Observables)
    % =========================================================================
    
    % Cell array format: {FunctionHandle, NameString}
    % Updated to 8 Periodic Test Functions (Fourier modes k=1, 2)
    funcs = {
        @(x) sin(x(:,1)),    'sin(x1)';
        @(x) cos(x(:,1)),    'cos(x1)';
        @(x) sin(2*x(:,1)),  'sin(2x1)';
        @(x) cos(2*x(:,1)),  'cos(2x1)';
        @(x) sin(x(:,2)),    'sin(x2)';
        @(x) cos(x(:,2)),    'cos(x2)';
        @(x) sin(2*x(:,2)),  'sin(2x2)';
        @(x) cos(2*x(:,2)),  'cos(2x2)'
    };
    
    num_funcs = size(funcs, 1);
    truth_vector = zeros(1, num_funcs);
    
    % Calculate Ground Truth Expectations E[f]
    for i = 1:num_funcs
        f_handle = funcs{i, 1};
        f_vals = f_handle(grid_points);
        % Numerical Integration: sum(f(x) * p(x))
        truth_vector(i) = sum(f_vals .* w_grid) / Z;
    end

    % =========================================================================
    % 4. Compute Sample Estimates
    % =========================================================================
    
    estimated_vector = zeros(1, num_funcs);
    num_samples = size(samples, 1);
    
    if isempty(weights)
        % ---- Case A: Unweighted (Standard MCMC) ----
        fprintf('[Analysis] Computing Unweighted Bias (N=%d)...\n', num_samples);
        
        for i = 1:num_funcs
            f_handle = funcs{i, 1};
            f_vals = f_handle(samples);
            estimated_vector(i) = mean(f_vals);
        end
    else
        % ---- Case B: Weighted (Importance Sampling) ----
        fprintf('[Analysis] Computing Weighted Bias (N=%d)...\n', num_samples);
        
        % Normalize weights to sum to 1
        w_sum = sum(weights);
        if w_sum == 0
            error('Error: Sum of importance weights is zero.');
        end
        weights_norm = weights / w_sum;
        
        for i = 1:num_funcs
            f_handle = funcs{i, 1};
            f_vals = f_handle(samples);
            % Weighted Average: sum(w_i * f(x_i))
            estimated_vector(i) = sum(f_vals .* weights_norm);
        end
    end

    % =========================================================================
    % 5. Compute L2 Bias and Display Results
    % =========================================================================
    
    % Calculate Error Vector
    bias_vector = estimated_vector - truth_vector;
    l2_bias = norm(bias_vector);
    
    % Formatting Output
    sep_line = repmat('-', 1, 70);
    
    fprintf('\n%s\n', sep_line);
    fprintf(' BIAS ANALYSIS REPORT (Beta = %.2f)\n', beta);
    fprintf('%s\n', sep_line);
    fprintf('%-15s | %-15s | %-15s | %-15s\n', 'Function', 'Estimate', 'Truth', 'Bias');
    fprintf('%s\n', sep_line);
    
    for i = 1:num_funcs
        name = funcs{i, 2};
        est = estimated_vector(i);
        truth = truth_vector(i);
        bias = bias_vector(i);
        fprintf('%-15s | %-15.6f | %-15.6f | %-15.6f\n', name, est, truth, bias);
    end
    
    fprintf('%s\n', sep_line);
    fprintf(' >> Overall L2 Bias: %.6e\n', l2_bias);
    fprintf('%s\n\n', sep_line);

end