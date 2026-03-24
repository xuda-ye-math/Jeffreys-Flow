classdef GMM_3D
    % GMM_3D Potential Energy System Class (Auto-Geometric Bridge)
    %
    % Logic:
    %   U_lambda(x) = (1 - lambda) * U_base(x) + lambda * U_target(x)
    %   Lambda Schedule: 0 -> [Geometric Progression] -> 1
    %
    %   The number of steps M is calculated automatically such that the
    %   ratio between adjacent non-zero lambdas never exceeds BETA_RATE.
    
    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 50000;
        MEAN        = 0;        
        SIGMA       = 2.5;      
        
        % ---- 2. Ladder Constraints ----
        % BETA_MIN: The starting value of the geometric section (Updated to 0.2)
        % BETA_RATE: The maximum allowed growth rate (ratio) between steps.
        BETA_MIN    = 0.08;      
        BETA_RATE   = 4;     
        
        % ---- 3. Derived Ladder Properties ----
        PT_M                    % Number of geometric intervals (Calculated)
        BETA_LIST               % [PT_M + 1, 1] Full lambda sequence including 0
        
        % ---- 4. Dynamics Parameters ----
        DT          = 5e-3;     
        COR_TIME    = 2;        
        CONST_SWAP  = 0.05;      
        WARM_TIME   = 200;      
        
        % ---- 5. Internal ----
        SWAP_INTERVAL_STEPS     
        TARGET_MEANS            
        TARGET_PRECISIONS       
        TARGET_LOG_DETS         
        TARGET_LOG_WEIGHTS      
    end
    
    properties (Constant)
        NAME = '3D Anisotropic GMM (Auto-Geometric)';
        ABBR = 'GM';
        DIM  = 3;
        
        X_LIM_COMPUTE = [-12, 12];
        Y_LIM_COMPUTE = [-12, 12];
        Z_LIM_COMPUTE = [-12, 12];
        
        X_LIM_PLOT    = [-8, 8];
        Y_LIM_PLOT    = [-8, 8];
        Z_LIM_PLOT    = [-8, 8];
    end
    
    methods
        function obj = GMM_3D(varargin)
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
            % We need a sequence start=BETA_MIN, end=1.0 with ratio <= BETA_RATE.
            % M >= 1 + log(1/start) / log(rate)
            
            numer = log(1.0 / obj.BETA_MIN);
            denom = log(obj.BETA_RATE);
            min_steps = ceil(1 + numer / denom);
            
            obj.PT_M = max(2, min_steps); % Ensure at least start and end
            
            % ---- 2. Construct Geometric Ladder ----
            % logspace(a, b, n) generates n points between 10^a and 10^b
            geom_part = logspace(log10(obj.BETA_MIN), log10(1.0), obj.PT_M)';
            
            % ---- 3. Full Ladder (Prepend 0) ----
            obj.BETA_LIST = [0; geom_part];
            
            % Dynamics
            swap_time = obj.CONST_SWAP * obj.COR_TIME;
            obj.SWAP_INTERVAL_STEPS = max(1, round(swap_time / obj.DT));
            
            obj = obj.init_target_parameters();
        end
        
        function obj = init_target_parameters(obj)
            % (Standard GMM Initialization - Unchanged)
            K = 6; L = 6.0;
            obj.TARGET_MEANS = [L,0,0; -L,0,0; 0,L,0; 0,-L,0; 0,0,L; 0,0,-L];
            obj.TARGET_PRECISIONS = zeros(K, 3, 3);
            obj.TARGET_LOG_DETS   = zeros(K, 1);
            
            shapes = [0.3,0.3,0.8; 0.3,0.8,0.3; 0.8,0.3,0.3; 0.3,0.6,0.6; 0.4,0.4,0.4; 0.25,0.8,0.25];
            angles = [0,0,0; 45,45,0; 30,0,30; 90,45,0; 15,15,15; 60,0,-60];
            
            for k = 1:K
                lambda_val = diag(shapes(k, :).^2);
                phi=deg2rad(angles(k,1)); theta=deg2rad(angles(k,2)); psi=deg2rad(angles(k,3));
                Rz = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
                Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
                Rx = [1 0 0; 0 cos(psi) -sin(psi); 0 sin(psi) cos(psi)];
                R = Rz * Ry * Rx;
                Sigma = R * lambda_val * R';
                obj.TARGET_PRECISIONS(k, :, :) = inv(Sigma);
                obj.TARGET_LOG_DETS(k) = log(det(Sigma));
            end
            weights = ones(K, 1) / K;
            obj.TARGET_LOG_WEIGHTS = log(weights);
        end
        
        function u = compute_potential(obj, x)
            u = obj.compute_potential_mixed(x, 1.0);
        end
        
        function u_mix = compute_potential_mixed(obj, x, lambda)
            % U_lam = (1-lam)U0 + lam*U1
            z = (x - obj.MEAN) ./ obj.SIGMA;
            u_base = 0.5 * sum(z.^2, 2);
            
            M_samples = size(x, 1); K = 6;
            log_probs = zeros(M_samples, K);
            const_term = -0.5 * obj.DIM * log(2 * pi);
            
            for k = 1:K
                diff = x - obj.TARGET_MEANS(k, :);
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                term = diff * P; 
                mahalanobis = -0.5 * sum(term .* diff, 2);
                log_gauss = const_term - 0.5 * obj.TARGET_LOG_DETS(k) + mahalanobis;
                log_probs(:, k) = obj.TARGET_LOG_WEIGHTS(k) + log_gauss;
            end
            max_log = max(log_probs, [], 2);
            log_p = max_log + log(sum(exp(log_probs - max_log), 2));
            u_target = -log_p;
            
            u_mix = (1.0 - lambda) .* u_base + lambda .* u_target;
        end
        
        function grad_mix = compute_gradient_mixed(obj, x, lambda)
            grad_base = (x - obj.MEAN) ./ (obj.SIGMA^2);
            
            M_samples = size(x, 1); K = 6;
            log_joint = zeros(M_samples, K);
            const_term = -0.5 * obj.DIM * log(2 * pi);
            
            for k = 1:K
                diff = x - obj.TARGET_MEANS(k, :);
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                mahalanobis = -0.5 * sum((diff * P) .* diff, 2);
                log_joint(:, k) = obj.TARGET_LOG_WEIGHTS(k) + const_term - 0.5 * obj.TARGET_LOG_DETS(k) + mahalanobis;
            end
            max_log = max(log_joint, [], 2);
            log_sum = max_log + log(sum(exp(log_joint - max_log), 2));
            responsibilities = exp(log_joint - log_sum); 
            
            grad_target = zeros(M_samples, 3);
            for k = 1:K
                diff = x - obj.TARGET_MEANS(k, :);
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                grad_comp = diff * P;
                grad_target = grad_target + responsibilities(:, k) .* grad_comp;
            end
            
            grad_mix = (1.0 - lambda) .* grad_base + lambda .* grad_target;
        end
        
        function x_next = integrator(obj, x)
            lambdas = obj.BETA_LIST; 
            dt      = obj.DT;
            grad = obj.compute_gradient_mixed(x, lambdas);
            
            raw_drift = -grad * dt;
            taming_factor = 1 + dt * abs(grad);
            tamed_drift = raw_drift ./ taming_factor;
            
            noise = randn(size(x));
            sigma = sqrt(2 * dt); 
            diffusion = sigma .* noise;
            
            x_next = x + tamed_drift + diffusion;
        end
        
        function grad = compute_gradient(obj, x)
            grad = obj.compute_gradient_mixed(x, 1.0);
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) ---\n', obj.NAME, obj.ABBR);
            fprintf('Strategy: Auto-Geometric Bridge (Rate <= %.2f)\n', obj.BETA_RATE);
            fprintf('Params: Min=%.3f, Max=1.0\n', obj.BETA_MIN);
            fprintf('Ladder Steps (M): %d (Geometric) -> Total Replicas: %d\n', obj.PT_M, length(obj.BETA_LIST));
            
            % [UPDATED]: Print Full Ladder
            fprintf('Lambda Sequence (Full): %s\n', mat2str(obj.BETA_LIST', 4));
            
            fprintf('Base Sigma: %.2f\n', obj.SIGMA);
            fprintf('----------------------------------\n');
        end
    end
end