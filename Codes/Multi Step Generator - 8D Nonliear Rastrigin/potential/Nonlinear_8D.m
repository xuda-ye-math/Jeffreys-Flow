classdef Nonlinear_8D
    % Nonlinear_8D Potential Energy System Class
    %
    % Target Distribution: Twisted Independent Rastrigin (No Inner Coupling)
    %
    % The potential is defined as a composition:
    %   $$ U_{total}(x) = U_{inner}(T(x)) $$
    %
    % 1. Outer Twist (Diffeomorphism) [KEPT]:
    %   $$ z = T(x) = x + \delta \cdot \tanh(x Q) $$
    %   where Q is a fixed orthogonal matrix and delta controls distortion.
    %
    % 2. Inner Potential (Decoupled Rastrigin) [MODIFIED]:
    %   $$ U_{inner}(z) = \sum_{i=1}^{D} [ 0.5*z_i^2 + A \cos(w z_i) ] $$
    %   (The coupling term sum(gamma * z_i * sin(z_{i+1})) is removed by setting gamma=0)
    %
    % Logic:
    %   $$ U_\lambda(x) = (1 - \lambda) U_{base}(x) + \lambda U_{target}(x) $$
    
    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 400000;
        
        % Base Distribution (Gaussian) Parameters
        MEAN        = 0;        
        SIGMA       = 1; 
        
        % ---- 2. Ladder Constraints ----
        BETA_MIN    = 0.02; 
        BETA_RATE   = 2;
        DELTA_MAX   = 0.05;
        
        % ---- 3. Derived Ladder Properties ----
        PT_M                    % Number of geometric intervals
        BETA_LIST               % [PT_M + 1, 1] Full lambda sequence
        
        % ---- 4. Dynamics Parameters ----
        DT          = 5e-3;     
        COR_TIME    = 1;        
        CONST_SWAP  = 0.002;      
        WARM_TIME   = 200;      
        
        % ---- 5. Internal ----
        SWAP_INTERVAL_STEPS
        
        % ---- 6. Potential Specific ----
        A_VAL       = 12;    % Amplitude A
        W_VAL       = 2;     % Frequency w
        
        % Distortion (Twist) Parameters
        DISTORT_DELTA = 1;   % Strength of the non-linear twist
        DISTORT_Q              % Orthogonal Matrix [D, D]
    end
    
    properties (Constant)
        NAME = '8D Nonlinear Rastrigin';
        ABBR = 'NR';
        DIM  = 8;
        
        % Visualization Limits (Projected)
        X_LIM_COMPUTE  = [-6, 6];
        X_LIM_PLOT    = [-4, 4];
    end
    
    methods
        function obj = Nonlinear_8D(varargin)
            % Initialize with fixed random seed for reproducibility of Matrix Q
            rng(3); 
            
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
                  
            % ---- 4. Initialize Distortion Matrix Q ----
            % Generate a random orthogonal matrix for mixing dimensions
            [Q_raw, ~] = qr(randn(obj.DIM));
            obj.DISTORT_Q = Q_raw;
        end
        
        function obj = init_target_parameters(obj)
            % No stochastic target parameters to initialize
        end
        
        function u = compute_potential(obj, x)
            % Helper to compute pure target potential (Beta=1)
            % Input: x [N, D]
            % Output: u [N, 1]
            u = obj.compute_potential_mixed(x, 1.0);
        end
        
        function [z, J_sech2] = map_forward(obj, x)
            % Applies the diffeomorphism T(x)
            % $$ z = x + \delta \tanh(x Q) $$
            % Input: x [N, D]
            % Output: 
            %   z [N, D] - Transformed coordinates
            %   J_sech2 [N, D] - Intermediate term for Jacobian (sech^2(xQ))
            
            pre_act = x * obj.DISTORT_Q;   % [N, D]
            act     = tanh(pre_act);       % [N, D]
            z       = x + obj.DISTORT_DELTA .* act; 
            
            if nargout > 1
                J_sech2 = 1.0 - act.^2;    % [N, D] (derivative of tanh is sech^2 = 1-tanh^2)
            end
        end
        
        function grad_x = map_gradient_backward(obj, grad_z, J_sech2)
            % Applies the chain rule for the twist: 
            % $$ \nabla_x U = \nabla_z U \cdot J_T $$
            % Vector-Jacobian Product:
            % $$ g_x = g_z + \delta (g_z \odot \text{sech}^2(xQ)) Q^T $$
            %
            % Input: 
            %   grad_z [N, D] - Gradient w.r.t z
            %   J_sech2 [N, D] - Cached sech^2 terms
            % Output:
            %   grad_x [N, D]
            
            term = grad_z .* J_sech2;      % Element-wise: [N, D]
            grad_x = grad_z + obj.DISTORT_DELTA .* (term * obj.DISTORT_Q');
        end

        function u_mix = compute_potential_mixed(obj, x, lambda)
            % Computes Interpolated Potential
            % Input: x [N, D], lambda [Scalar]
            % Output: u_mix [N, 1]
            
            % 1. Base Potential (Isotropic Gaussian on raw x)
            z_base = (x - obj.MEAN) ./ obj.SIGMA;
            u_base = 0.5 * sum(z_base.^2, 2);
            
            % 2. Target Potential (Twisted Decoupled Rastrigin)
            % First, apply the Twist T(x)
            z = obj.map_forward(x);  % z is [N, D]
            
            % Then compute Rastrigin on z (No coupling)
            % Base Rastrigin: 0.5*z^2 + A*cos(w*z)
            term_rastrigin = sum(0.5 * z.^2 + obj.A_VAL .* cos(obj.W_VAL .* z), 2);
                  
            u_target = term_rastrigin;
            
            % 3. Mix
            u_mix = (1.0 - lambda) .* u_base + lambda .* u_target;
        end
        
        function grad_mix = compute_gradient_mixed(obj, x, lambda)
            % Computes Interpolated Gradient
            % Input: x [N, D], lambda [Scalar or Vector]
            % Output: grad_mix [N, D]
            
            % 1. Base Gradient
            grad_base = (x - obj.MEAN) ./ (obj.SIGMA^2);
            
            % 2. Target Gradient (Twisted)
            % A. Forward Map to get z and Jacobian parts
            [z, J_sech2] = obj.map_forward(x);
            
            % B. Compute Gradient w.r.t z (Inner Potential)
            grad_inner = zeros(size(z));
            
            % Rastrigin Part w.r.t z
            grad_inner = grad_inner + z - obj.A_VAL * obj.W_VAL * sin(obj.W_VAL .* z);
            
            % C. Backward Map (Chain Rule) to get Gradient w.r.t x
            grad_target = obj.map_gradient_backward(grad_inner, J_sech2);
            
            % 3. Mix
            grad_mix = (1.0 - lambda) .* grad_base + lambda .* grad_target;
        end
        
        function x_next = integrator(obj, x)
            % Overdamped Langevin Dynamics
            % Input: x [N, D]
            % Output: x_next [N, D]
            lambdas = obj.BETA_LIST; 
            dt      = obj.DT;
            
            grad = obj.compute_gradient_mixed(x, lambdas);
            
            % Tamed Euler-Maruyama
            raw_drift = -grad * dt;
            taming_factor = 1 + dt * abs(grad); 
            tamed_drift = raw_drift ./ taming_factor;
            
            noise = randn(size(x));
            sigma = sqrt(2 * dt); 
            diffusion = sigma .* noise;
            
            x_next = x + tamed_drift + diffusion;
        end
        
        function grad = compute_gradient(obj, x)
            % Helper for pure target gradient
            % Input: x [N, D]
            % Output: grad [N, D]
            grad = obj.compute_gradient_mixed(x, 1.0);
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d\n', obj.DIM);
            fprintf('Strategy: Auto-Geometric Bridge (Rate <= %.2f)\n', obj.BETA_RATE);
            fprintf('Params: Min=%.3f, Max=1.0\n', obj.BETA_MIN);
            fprintf('Twist Map: x + %.2f * tanh(x * Q)\n', obj.DISTORT_DELTA);
            fprintf('----------------------------------\n');
            
            % Save Distortion Matrix
            if ~exist('data', 'dir'), mkdir('data'); end
            Q = obj.DISTORT_Q;
            save('data/DISTORT_Q.mat', 'Q');
            fprintf('>> Exported DISTORT_Q to data/DISTORT_Q.mat\n');
        end
    end
end