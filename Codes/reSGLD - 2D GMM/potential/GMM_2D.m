classdef GMM_2D
    % GMM_2D Potential Energy System Class
    %
    % Logic:
    %   U_lambda(x) = (1 - lambda) * U_base(x) + lambda * U_target(x)
    %   Lambda Schedule: 0 -> [Geometric Progression] -> 1
    %
    %   The number of steps M is calculated automatically such that the
    %   ratio between adjacent non-zero lambdas never exceeds BETA_RATE.
    
    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 40000;
        MEAN        = 0;        
        SIGMA       = 1.5;
        NOISE       = 4.0;
        
        % ---- 3. Derived Ladder Properties ----
        PT_M = 5;    % Number of geometric intervals
        BETA_LIST = [0; 0.05; 0.2; 0.4; 0.7; 1]    % [PT_M + 1, 1] Full lambda sequence including 0
        
        % ---- 4. Dynamics Parameters ----
        DT          = 1e-2;     
        COR_TIME    = 2;        
        CONST_SWAP  = 0.02;      
        WARM_TIME   = 200;      
        
        % ---- 5. Internal ----
        SWAP_INTERVAL_STEPS     
        TARGET_MEANS            
        TARGET_PRECISIONS       
        TARGET_LOG_DETS         
        TARGET_LOG_WEIGHTS      
    end
    
    properties (Constant)
        NAME = '2D Anisotropic GMM';
        ABBR = 'GM';
        DIM  = 2; % Updated to 2D
        
        X_LIM_COMPUTE = [-10, 10];
        Y_LIM_COMPUTE = [-10, 10];
        
        X_LIM_PLOT    = [-2.2, 2.2];
        Y_LIM_PLOT    = [-2.2, 2.2];
    end
    
    methods
        function obj = GMM_2D(varargin)
            p = inputParser;
            addParameter(p, 'MU_SIZE', obj.MU_SIZE);
            addParameter(p, 'MEAN', obj.MEAN);
            addParameter(p, 'SIGMA', obj.SIGMA);
            addParameter(p, 'BETA_LIST', obj.BETA_LIST);
            addParameter(p, 'DT', obj.DT);
            addParameter(p, 'COR_TIME', obj.COR_TIME);
            addParameter(p, 'CONST_SWAP', obj.CONST_SWAP);
            addParameter(p, 'WARM_TIME', obj.WARM_TIME);
            
            parse(p, varargin{:});
            fields = fieldnames(p.Results);
            for i = 1:length(fields), obj.(fields{i}) = p.Results.(fields{i}); end   
         
            % Dynamics
            swap_time = obj.CONST_SWAP * obj.COR_TIME;
            obj.SWAP_INTERVAL_STEPS = max(1, round(swap_time / obj.DT));
            
            obj = obj.init_target_parameters();
        end
        
        function obj = init_target_parameters(obj)

            % 4 irregular wells as specified
            % Weights
            weights = [0.25; 0.3; 0.15; 0.3];
            
            % Means (Centers)
            obj.TARGET_MEANS = [ ...
                -1.2, -1.2; ... % Bottom Left
                 1.3, -0.8; ... % Bottom Right
                -0.5,  1.4; ... % Top Left
                 1.2,  1.3      % Top Right
            ];
            
            % Covariances (Shapes)
            covs = cat(3, ...
                [0.06, 0.02; 0.02, 0.06], ...      % Well 1
                [0.08, -0.04; -0.04, 0.12], ...    % Well 2
                [0.05, 0.0; 0.0, 0.03], ...        % Well 3
                [0.04, 0.01; 0.01, 0.04]);         % Well 4
            
            K = length(weights);
            obj.TARGET_PRECISIONS = zeros(K, obj.DIM, obj.DIM);
            obj.TARGET_LOG_DETS   = zeros(K, 1);
            
            for k = 1:K
                Sigma = covs(:, :, k);
                % Precision is the inverse of Covariance
                obj.TARGET_PRECISIONS(k, :, :) = inv(Sigma);
                % Store log determinant for PDF calculation
                obj.TARGET_LOG_DETS(k) = log(det(Sigma));
            end
            
            % Log weights for numerical stability in LogSumExp
            obj.TARGET_LOG_WEIGHTS = log(weights);
        end
        
        function u = compute_potential(obj, x, i_vec)
            u = obj.compute_potential_mixed(x, 1.0, i_vec);
        end
        
        function u_mix = compute_potential_mixed(obj, x, lambda, i_vec)
            % Input x: [N_samples x 2]
            % U_lam = (1-lam)U0 + lam*U1
            
            % Base Potential (Gaussian centered at MEAN with SIGMA)
            z = (x - obj.MEAN) ./ obj.SIGMA;
            u_base = 0.5 * sum(z.^2, 2);
            
            % Target Potential (GMM)
            % U_target(x) = -log( sum( w_k * N(x|mu_k, Sig_k) ) )
            M_samples = size(x, 1);
            K = size(obj.TARGET_MEANS, 1);
            log_probs = zeros(M_samples, K);
            const_term = -0.5 * obj.DIM * log(2 * pi);
            
            for k = 1:K
                % Difference vector: (x - mu_k)
                diff = x - obj.TARGET_MEANS(k, :); 
                
                % Precision matrix for k-th component
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                
                % Mahalanobis distance: 0.5 * (x-mu)^T * P * (x-mu)
                % Vectorized: sum( (diff*P) .* diff )
                term = diff * P; 
                mahalanobis = -0.5 * sum(term .* diff, 2);
                
                % Log PDF for component k
                log_gauss = const_term - 0.5 * obj.TARGET_LOG_DETS(k) + mahalanobis;
                log_probs(:, k) = obj.TARGET_LOG_WEIGHTS(k) + log_gauss;
            end
            
            % Log-Sum-Exp Trick: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
            max_log = max(log_probs, [], 2);
            log_p = max_log + log(sum(exp(log_probs - max_log), 2));
            u_target = -log_p;
            
            % Random Potential
            if any(i_vec ~= 0)
                u_rand = i_vec(1) .* cos(x(:, 1)) + i_vec(2) .* cos(x(:, 2));
            else
                u_rand = 0;
            end
            
            % Mixed Potential
            u_mix = (1.0 - lambda) .* u_base + lambda .* u_target + u_rand;
        end
        
        function grad_mix = compute_gradient_mixed(obj, x, lambda, i_vec)
            % Gradient of Base Potential
            % grad U_base = (x - mean) / sigma^2
            grad_base = (x - obj.MEAN) ./ (obj.SIGMA^2);
            
            % Gradient of Target Potential (GMM)
            M_samples = size(x, 1);
            K = size(obj.TARGET_MEANS, 1);
            log_joint = zeros(M_samples, K);
            const_term = -0.5 * obj.DIM * log(2 * pi);
            
            % 1. Calculate Responsibilities (Posterior probabilities of components)
            for k = 1:K
                diff = x - obj.TARGET_MEANS(k, :);
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                mahalanobis = -0.5 * sum((diff * P) .* diff, 2);
                log_joint(:, k) = obj.TARGET_LOG_WEIGHTS(k) + const_term - 0.5 * obj.TARGET_LOG_DETS(k) + mahalanobis;
            end
            
            max_log = max(log_joint, [], 2);
            log_sum = max_log + log(sum(exp(log_joint - max_log), 2));
            % gamma_k = P(k|x)
            responsibilities = exp(log_joint - log_sum); 
            
            % 2. Calculate Weighted Gradients
            % grad U_target = sum( gamma_k * P_k * (x - mu_k) )
            grad_target = zeros(M_samples, obj.DIM);
            for k = 1:K
                diff = x - obj.TARGET_MEANS(k, :);
                P = squeeze(obj.TARGET_PRECISIONS(k, :, :));
                
                % Component gradient part: Sigma^-1 * (x - mu)
                grad_comp = diff * P;
                
                % Accumulate weighted by responsibility
                grad_target = grad_target + responsibilities(:, k) .* grad_comp;
            end
            
            % Random Gradient
            if any(i_vec ~= 0)
                % Ensure broadcast if i_vec is 1xDIM; otherwise handle size
                % Assuming i_vec is [i_1, i_2] and x is [N, 2]
                grad_rand = -i_vec .* sin(x); 
            else
                grad_rand = 0;
            end
            
            % Mixed Gradient
            grad_mix = (1.0 - lambda) .* grad_base + lambda .* grad_target + grad_rand;
        end
        
        function x_next = integrator(obj, x, i_vec)
            if nargin < 3
                i_vec = [0, 0];
            end
            
            lambdas = obj.BETA_LIST; 
            dt      = obj.DT;
            grad = obj.compute_gradient_mixed(x, lambdas, i_vec);
            
            raw_drift = -grad * dt;
            taming_factor = 1 + dt * abs(grad);
            tamed_drift = raw_drift ./ taming_factor;
            
            noise = randn(size(x));
            sigma = sqrt(2 * dt); 
            diffusion = sigma .* noise;
            
            x_next = x + tamed_drift + diffusion;
        end
        
        function grad = compute_gradient(obj, x, i_vec)
            grad = obj.compute_gradient_mixed(x, 1.0, i_vec);
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) ---\n', obj.NAME, obj.ABBR);
            fprintf('Ladder Steps (M): %d (Geometric) -> Total Replicas: %d\n', obj.PT_M, length(obj.BETA_LIST));
            
            fprintf('Lambda Sequence (Full): %s\n', mat2str(obj.BETA_LIST, 4));
            fprintf('Base Sigma: %.2f\n', obj.SIGMA);
            fprintf('Target: 4-Well Irregular GMM (Energy Gap ~20)\n');
            fprintf('----------------------------------\n');
        end
    end
end