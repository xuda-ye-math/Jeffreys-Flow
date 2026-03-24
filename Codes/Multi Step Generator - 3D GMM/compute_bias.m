function l2_bias = compute_bias(para, samples, beta, weights)
    % COMPUTE_BIAS Computes the L2 bias of 3D samples against ground truth.
    % Uses analytical expectations for the Target distribution (Beta=1)
    % based on the GMM parameters stored in the 'para' object.
    
    if nargin < 4, weights = []; end
    if ~isempty(weights), weights = weights(:); end
    
    % Check dimensions
    if size(samples, 2) ~= 3
        error('Samples must be 3D [N, 3].');
    end

    % Define Test Functions (6 functions: sin/cos for x, y, z)
    funcs = {
        @(x) sin(x(:,1));
        @(x) cos(x(:,1));
        @(x) sin(x(:,2));
        @(x) cos(x(:,2));
        @(x) sin(x(:,3));
        @(x) cos(x(:,3))
    };
    num_funcs = length(funcs);
    truth_vector = zeros(1, num_funcs);

    % =========================================================================
    % 1. Compute Analytical Ground Truth (Target GMM)
    % =========================================================================
    
    % Verify Beta is 1 (Target)
    if abs(beta - 1.0) > 1e-3
        warning('Analytical bias computation in compute_bias is strictly valid for Beta=1.0 (Target). Current Beta=%.2f.', beta);
    end

    % Retrieve GMM parameters directly from the para object
    % Ensure GMM_3D class has initialized these properties
    if ~isprop(para, 'TARGET_MEANS')
        error('The para object does not have TARGET_MEANS property. Ensure GMM_3D is initialized correctly.');
    end

    means = para.TARGET_MEANS;             % [K, 3]
    precs = para.TARGET_PRECISIONS;        % [K, 3, 3]
    log_weights = para.TARGET_LOG_WEIGHTS; % [K, 1]
    
    % Calculate normalized mixture weights
    w_k = exp(log_weights);
    w_k = w_k / sum(w_k);
    K = length(w_k);

    % Pre-compute marginal variances (diagonal of Sigma) for each component
    % E[sin(x_d)] = exp(-sigma_{k,d}^2 / 2) * sin(mu_{k,d})
    sigmas_sq = zeros(K, 3);
    for k = 1:K
        P = squeeze(precs(k, :, :));
        Sigma = inv(P);
        sigmas_sq(k, :) = diag(Sigma);
    end

    % Compute Truth Vector
    for i = 1:num_funcs
        % Determine function type (sin/cos) and dimension (1,2,3)
        % i=1: sin(x1), i=2: cos(x1), i=3: sin(x2)...
        is_sin = mod(i, 2) == 1; 
        dim_idx = floor((i-1)/2) + 1;
        
        val_true = 0;
        for k = 1:K
            mu_val = means(k, dim_idx);
            var_val = sigmas_sq(k, dim_idx);
            
            % Characteristic function decay factor
            decay = exp(-0.5 * var_val);
            
            if is_sin
                term = decay * sin(mu_val);
            else
                term = decay * cos(mu_val);
            end
            
            val_true = val_true + w_k(k) * term;
        end
        truth_vector(i) = val_true;
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