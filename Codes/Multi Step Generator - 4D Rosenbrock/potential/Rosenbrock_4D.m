classdef Rosenbrock_4D
    % ROSENBROCK_4D Potential Energy System Class
    %
    % Target Distribution:
    %   $$ U(x) = 6 \sum_{i=1}^{D-1} [ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ] $$
    %   Dimension D = 4.
    %
    % Logic:
    %   $$ U_\lambda(x) = (1 - \lambda) U_{base}(x) + \lambda U_{target}(x) $$
    
    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 100000;
        
        % Base Distribution (Gaussian) Parameters
        % Vectorized for D=4
        MEAN        = [0, 0.6, 1, 2];        
        SIGMA       = [1.0, 1.0, 1.5, 2]; 
        
        % ---- 2. Ladder Constraints ----
        BETA_MIN    = 0.001; % Updated from log output     
        BETA_RATE   = 2;     
        
        % ---- 3. Derived Ladder Properties ----
        PT_M                    % Number of geometric intervals
        BETA_LIST               % [PT_M + 1, 1] Full lambda sequence
        
        % ---- 4. Dynamics Parameters ----
        DT          = 1e-4;     % Reduced slightly for stability with Rosenbrock
        COR_TIME    = 1;        
        CONST_SWAP  = 0.005;      
        WARM_TIME   = 200;      
        
        % ---- 5. Internal ----
        SWAP_INTERVAL_STEPS     
    end
    
    properties (Constant)
        NAME = '4D Rosenbrock (Scaled)';
        ABBR = 'RB';
        DIM  = 4;
        
        % Visualization Limits (Projected)
        X1_LIM_COMPUTE = [-1.6, 2];
        X2_LIM_COMPUTE = [-0.5, 2];
        X3_LIM_COMPUTE = [-0.5, 3];
        X4_LIM_COMPUTE = [-0.5, 6.5];
        
        X1_LIM_PLOT    = [-1.3, 1.4];
        X2_LIM_PLOT    = [-0.1, 1.6];
        X3_LIM_PLOT    = [-0.1, 2.5];
        X4_LIM_PLOT    = [-1, 6.0];
    end
    
    methods
        function obj = Rosenbrock_4D(varargin)
            p = inputParser;
            addParameter(p, 'MU_SIZE', obj.MU_SIZE);
            addParameter(p, 'MEAN', obj.MEAN);
            addParameter(p, 'SIGMA', obj.SIGMA);
            addParameter(p, 'BETA_MIN', obj.BETA_MIN);
            addParameter(p, 'BETA_RATE', obj.BETA_RATE);
            addParameter(p, 'DT', obj.DT);
            addParameter(p, 'COR_TIME', obj.COR_TIME);
            addParameter(p, 'CONST_SWAP', obj.CONST_SWAP);
            addParameter(p, 'WARM_TIME', obj.WARM_TIME);
            
            parse(p, varargin{:});
            fields = fieldnames(p.Results);
            for i = 1:length(fields), obj.(fields{i}) = p.Results.(fields{i}); end
            
            % [FIX] Ensure Row Vectors for safe broadcasting
            obj.MEAN = obj.MEAN(:)';
            obj.SIGMA = obj.SIGMA(:)';
            
            % ---- 1. Calculate Minimum M ----
            numer = log(1.0 / obj.BETA_MIN);
            denom = log(obj.BETA_RATE);
            min_steps = ceil(1 + numer / denom);
            obj.PT_M = max(2, min_steps); 
            
            % ---- 2. Construct Geometric Ladder ----
            geom_part = logspace(log10(obj.BETA_MIN), log10(1.0), obj.PT_M)';
            obj.BETA_LIST = [0; geom_part];
            
            % Dynamics
            swap_time = obj.CONST_SWAP * obj.COR_TIME;
            obj.SWAP_INTERVAL_STEPS = max(1, round(swap_time / obj.DT));
            
            % No GMM initialization required
        end
        
        function obj = init_target_parameters(obj)
            % No target parameters to initialize for Rosenbrock
        end
        
        function u = compute_potential(obj, x)
            % Helper to compute pure target potential (Beta=1)
            % Input: x [N, 4]
            % Output: u [N, 1]
            u = obj.compute_potential_mixed(x, 1.0);
        end
        
        function u_mix = compute_potential_mixed(obj, x, lambda)
            % Computes Interpolated Potential:
            % $$ U_{mix} = (1-\lambda)U_{base} + \lambda U_{target} $$
            %
            % Input: 
            %   x      : [N, 4] Coordinates
            %   lambda : [1] Scalar interpolation factor (or vector [N, 1])
            % Output:
            %   u_mix  : [N, 1] Potential Energy
            
            % 1. Base Potential (Gaussian)
            % U_base = 0.5 * ||(x - mu)/sigma||^2
            % [FIX] Use broadcasting for vector MEAN/SIGMA
            z = (x - obj.MEAN) ./ obj.SIGMA;
            u_base = 0.5 * sum(z.^2, 2);
            
            % 2. Target Potential (Scaled Rosenbrock 4D)
            % U(x) = sum_{i=1}^{3} [ 250*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
            scale_factor = 1;
            term_sum = zeros(size(x, 1), 1);
            
            % Loop over dimensions 1 to D-1
            for i = 1:(obj.DIM - 1)
                x_i = x(:, i);
                x_next = x(:, i+1);
                
                term = 200 * (x_next - x_i.^2).^2 + (1 - x_i).^2;
                term_sum = term_sum + term;
            end
            
            u_target = scale_factor * term_sum;
            
            % 3. Mix
            u_mix = (1.0 - lambda) .* u_base + lambda .* u_target;
        end
        
        function grad_mix = compute_gradient_mixed(obj, x, lambda)
            % Computes Interpolated Gradient:
            % $$ \nabla U_{mix} = (1-\lambda)\nabla U_{base} + \lambda \nabla U_{target} $$
            %
            % Input:
            %   x      : [N, 4] Coordinates
            %   lambda : [1] Scalar or [N, 1] Vector
            % Output:
            %   grad_mix : [N, 4] Gradient vectors
            
            % 1. Base Gradient
            % [FIX] Use element-wise power (.^) for vector SIGMA
            grad_base = (x - obj.MEAN) ./ (obj.SIGMA.^2);
            
            % 2. Target Gradient (Rosenbrock)
            % We accumulate gradients for each term in the summation
            grad_target = zeros(size(x));
            scale_factor = 4.0;
            
            % Term i involves x_i and x_{i+1}
            for i = 1:(obj.DIM - 1)
                x_i = x(:, i);
                x_next = x(:, i+1);
                
                % Common factor
                diff = (x_next - x_i.^2);
                
                % d/d(x_i) of [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
                % = 200 * diff * (-2*x_i) + 2 * (1 - x_i) * (-1)
                d_xi = -400 * x_i .* diff - 2 * (1 - x_i);
                
                % d/d(x_{i+1}) of [100(x_{i+1} - x_i^2)^2]
                % = 200 * diff
                d_xnext = 200 * diff;
                
                % Accumulate
                grad_target(:, i)   = grad_target(:, i)   + d_xi;
                grad_target(:, i+1) = grad_target(:, i+1) + d_xnext;
            end
            
            grad_target = grad_target * scale_factor;
            
            % 3. Mix
            grad_mix = (1.0 - lambda) .* grad_base + lambda .* grad_target;
        end
        
        function x_next = integrator(obj, x)
            % Overdamped Langevin Dynamics
            lambdas = obj.BETA_LIST; 
            dt      = obj.DT;
            
            % Compute Gradient at current positions for specific lambdas
            % Note: In simulate_MU.m, x is [Replicas, Dim], lambdas is [Replicas, 1]
            grad = obj.compute_gradient_mixed(x, lambdas);
            
            % Tamed Euler-Maruyama
            raw_drift = -grad * dt;
            taming_factor = 1 + dt * abs(grad); % Coordinate-wise taming
            tamed_drift = raw_drift ./ taming_factor;
            
            noise = randn(size(x));
            sigma = sqrt(2 * dt); 
            diffusion = sigma .* noise;
            
            x_next = x + tamed_drift + diffusion;
        end
        
        function grad = compute_gradient(obj, x)
            % Helper for pure target gradient
            grad = obj.compute_gradient_mixed(x, 1.0);
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d\n', obj.DIM);
            fprintf('Strategy: Auto-Geometric Bridge (Rate <= %.2f)\n', obj.BETA_RATE);
            fprintf('Params: Min=%.3f, Max=1.0\n', obj.BETA_MIN);
            fprintf('Ladder Steps (M): %d -> Total Replicas: %d\n', obj.PT_M, length(obj.BETA_LIST));
            
            % Handle vector printing for Sigma
            if length(obj.SIGMA) > 1
                fprintf('Base Sigma (Vector): [%s]\n', num2str(obj.SIGMA, '%.2f '));
            else
                fprintf('Base Sigma: %.2f\n', obj.SIGMA);
            end
            fprintf('----------------------------------\n');
        end
    end
end