function grad_U_mix = get_gradient_mixed_partial(theta, solver, y_obs, obs_indices, sigma_noise, batch_indices, beta)
% GET_GRADIENT_MIXED_PARTIAL Computes the unbiased mini-batch estimator
% of the gradient of the mixed potential U_mix for reSGLD.
%
% This function utilizes the Adjoint State Method.
% Formulation: grad_U_mix(x) = beta * grad_U_target(x)
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
%   grad_U_mix    - [8 x 1] Unbiased gradient estimator of the mixed potential

    % Total number of available observations (should be 81 in our setup)
    num_total_obs = length(y_obs);
    num_batch_obs = length(batch_indices);
    
    % Validation
    if max(batch_indices) > num_total_obs || min(batch_indices) < 1
        error('Batch indices must be within the range [1, %d].', num_total_obs);
    end

    % 1. Forward propagation: Solve Au = f
    u_full = solver.solve_forward(theta);
    
    % 2. Extract batch states and compute the gradient of the potential w.r.t the state u
    % This is the vector 'v' in the adjoint equation: A^T * lambda = v
    % Since A is symmetric (5-point Laplacian), this is just A * lambda = v
    
    batch_grid_indices = obs_indices(batch_indices);
    G_theta_batch = u_full(batch_grid_indices);
    y_obs_batch = y_obs(batch_indices);
    
    % Residual: (G(theta) - y_obs)
    residual_batch = G_theta_batch(:) - y_obs_batch(:);
    
    % Scaling factor for unbiased batch estimator
    scale_factor = num_total_obs / num_batch_obs;
    
    % The derivative of the potential U with respect to u at the observation points
    % dv_batch = (scale_factor / sigma_noise^2) * residual_batch
    % And incorporate beta here directly: dv_batch_mixed = beta * dv_batch
    dv_batch_mixed = beta * (scale_factor / sigma_noise^2) * residual_batch;
    
    % Construct the full right-hand side vector v (size N x 1)
    v = zeros(solver.N, 1);
    v(batch_grid_indices) = dv_batch_mixed;
    
    % 3. Adjoint State Equation: Solve A * lambda = v
    % We use the same pre-assembled sparse matrix solver.A
    lambda = solver.A \ v;
    
    % 4. Final Gradient Assembly: grad = (df/dtheta)^T * lambda
    grad_U_mix = solver.compute_gradient_f_theta(theta, lambda);

end
