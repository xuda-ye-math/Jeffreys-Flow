function l2_bias = compute_bias(para, samples, beta, weights)
    % COMPUTE_BIAS Computes the L2 bias of 2D samples against ground truth.
    % Uses numerical integration on a fine 2000x2000 grid for the Target
    % distribution (Beta=beta, default 1) based on the potential provided by 'para'.
    %
    % Test Functions: sin(x_i), cos(x_i), tanh(x_i) for i = 1, 2
    
    if nargin < 4, weights = []; end
    if ~isempty(weights), weights = weights(:); end
    
    % Check dimensions
    if size(samples, 2) ~= 2
        error('Samples must be 2D [N, 2].');
    end

    % Define Test Functions (6 functions)
    % Order: sin(x1), cos(x1), tanh(x1), sin(x2), cos(x2), tanh(x2)
    funcs = {
        @(x) sin(x(:,1));
        @(x) cos(x(:,1));
        @(x) tanh(x(:,1));
        @(x) sin(x(:,2));
        @(x) cos(x(:,2));
        @(x) tanh(x(:,2))
    };
    num_funcs = length(funcs);
    truth_vector = zeros(1, num_funcs);

    % =========================================================================
    % 1. Compute Numerical Ground Truth (Target GMM mixed with Base)
    % =========================================================================
    
    % Use beta for the potential calculation (0 to 1)
    if isempty(beta), beta = 1.0; end

    % Define Grid (2000 x 2000)
    N_GRID = 2000;
    
    % Use limits from para object
    x_min = para.X_LIM_COMPUTE(1);
    x_max = para.X_LIM_COMPUTE(2);
    y_min = para.Y_LIM_COMPUTE(1);
    y_max = para.Y_LIM_COMPUTE(2);
    
    x_vec = linspace(x_min, x_max, N_GRID);
    y_vec = linspace(y_min, y_max, N_GRID);
    [X, Y] = meshgrid(x_vec, y_vec);
    grid_pts = [X(:), Y(:)];
    
    % Compute Potential on Grid (Target: lambda=beta)
    % Note: compute_potential_mixed handles [N, 2] input
    % [UPDATED] Pass [0, 0] for i_vec (Accurate calculation)
    u_vals = para.compute_potential_mixed(grid_pts, beta, [0, 0]);
    
    % Compute Normalized Probability Density
    % P(x) = exp(-U(x)) / Z
    % Use log-sum-exp trick for stability
    log_probs = -u_vals;
    max_log = max(log_probs);
    probs = exp(log_probs - max_log);
    
    % Normalize: sum(probs * dx * dy) = 1
    % Since dx*dy is constant, we can just normalize the discrete sum to 1
    % effectively treating it as a PMF on the grid points for expectation calculation
    probs = probs / sum(probs);
    
    % Compute Truth Vector (Expectations on Grid)
    for i = 1:num_funcs
        f = funcs{i};
        f_vals = f(grid_pts);
        truth_vector(i) = sum(f_vals .* probs);
    end

    % =========================================================================
    % 2. Compute Sample Estimates
    % =========================================================================
    estimated_vector = zeros(1, num_funcs);
    
    if isempty(weights)
        % Unweighted (Reference Samples)
        for i = 1:num_funcs
            f = funcs{i};
            estimated_vector(i) = mean(f(samples));
        end
    else
        % Weighted (Generated Samples)
        w_norm = weights / sum(weights);
        for i = 1:num_funcs
            f = funcs{i};
            estimated_vector(i) = sum(f(samples) .* w_norm);
        end
    end

    % =========================================================================
    % 3. Compute L2 Bias
    % =========================================================================
    diff = estimated_vector - truth_vector;
    l2_bias = norm(diff);
end
