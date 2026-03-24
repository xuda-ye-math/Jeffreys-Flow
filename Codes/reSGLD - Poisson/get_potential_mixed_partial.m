function U_mix = get_potential_mixed_partial(theta, solver, y_obs, obs_indices, sigma_noise, batch_indices, beta)
% GET_POTENTIAL_MIXED_PARTIAL Computes the mixed unbiased mini-batch estimator
% of the potential U_mix for reSGLD across a specific stochastic sensor subset.
%
% Formulation: U_mix(x) = (1 - beta) * U_base(x) + beta * U_target(x)
%
% Since the base distribution p_0(theta) is a uniform distribution,
% its negative log-likelihood U_base(theta) is constant. Therefore, we 
% simply scale the partial target log-likelihood computation by beta.
%
% Inputs:
%   theta         - [8 x 1] vector of the 4 source coordinates
%   solver        - Poisson_2D object, pre-initialized with grid parameters
%   y_obs         - [81 x 1] vector of the noisy synthetic observations
%   obs_indices   - [81 x 1] vector of absolute linear indices mapping to the
%                   full state solution u
%   sigma_noise   - Scalar value for the std of observation noise
%   batch_indices - [batch_size x 1] vector of INDICES inside the 81 sensors
%                   (i.e., values ranging from 1 to 81) indicating which sensors
%                   to evaluate in this SGLD step.
%   beta          - Inverse temperature [0, 1]. beta=0 is prior, beta=1 is pure posterior.
%
% Outputs:
%   U_mix         - Unbiased scalar estimator of the mixed potential (scaled by beta)

    % Total number of available observations (should be 81 in our setup)
    num_total_obs = length(y_obs);
    num_batch_obs = length(batch_indices);
    
    % Validation
    if max(batch_indices) > num_total_obs || min(batch_indices) < 1
        error('Batch indices must be within the range [1, %d].', num_total_obs);
    end

    % 1. Solve the forward problem to get the full state u
    u_full = solver.solve_forward(theta);
    
    % 2. Extract ONLY the batch-specific states from the full grid
    batch_grid_indices = obs_indices(batch_indices);
    G_theta_batch = u_full(batch_grid_indices);
    
    % Ensure representations match
    G_theta_batch = G_theta_batch(:);
    y_obs_batch = y_obs(batch_indices);
    y_obs_batch = y_obs_batch(:);
    
    % 3. Calculate the sum of squared residuals for the batch
    residual_batch = G_theta_batch - y_obs_batch;
    ssr_batch = sum(residual_batch.^2);
    
    % 4. Compute the basic potential for the batch
    U_batch = ssr_batch / (2 * sigma_noise^2);
    
    % 5. Scale the potential to make it an unbiased estimator of the full target log-likelihood
    scale_factor = num_total_obs / num_batch_obs;
    U_target_partial = U_batch * scale_factor;
    
    % 6. Scale by beta to get the mixed potential (U_base is 0)
    U_mix = beta * U_target_partial;
end
