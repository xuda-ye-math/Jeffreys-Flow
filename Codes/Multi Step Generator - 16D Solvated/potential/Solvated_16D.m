classdef Solvated_16D
    % Solvated_16D Potential Energy System Class
    %
    % Target Distribution: Solvated Particle in a Periodic Potential
    % Manifold: T^2 x R^14  (First 2 dims periodic, 14 dims Euclidean)
    %
    % U(x) = U_grid(x1, x2) + U_bath(x1, x2, x3:16)
    %
    % 1. Periodic Grid (4x4 Wells):
    %   U_grid = A * [ cos(4*x1) + cos(4*x2) ]
    %   (x1, x2 in [-pi, pi])
    %
    % 2. Harmonic Solvent Coupling:
    %   U_bath = sum_{k=3}^{16} 0.5 * k_s * (x_k - alpha * sin(w*x1)*sin(w*x2))^2
    
    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 500000;
        
        % Base Distribution (Gaussian) Parameters
        MEAN        = 0;        
        SIGMA       = 1; 
        
        % ---- 2. Ladder Constraints ----
        BETA_MIN    = 0.001; 
        BETA_RATE   = 1.5;
        DELTA_MAX   = 0.05;
        
        % ---- 3. Derived Ladder Properties ----
        PT_M                    % Number of geometric intervals
        BETA_LIST               % [PT_M + 1, 1] Full lambda sequence
        
        % ---- 4. Dynamics Parameters ----
        DT          = 1e-3;     % Smaller DT for stiff constraints
        COR_TIME    = 1;        
        CONST_SWAP  = 0.002;      
        WARM_TIME   = 200;      
        
        % ---- 5. Internal ----
        SWAP_INTERVAL_STEPS
        
        % ---- 6. Potential Specific ----
        % Grid Parameters (4x4 in [-pi, pi])
        A_GRID      = 2.0;   % Barrier height for grid
        FREQ_GRID   = 4.0;   % Frequency 4 implies 4 wells in 2pi (if cos(4x))
        
        % Coupling Parameters
        K_SPRING    = 30.0;  % Stiff solvent (Constant)
        
        ALPHA_C     % Now a vector [1, 14] or similar
        ALPHA_MIN   = 2.0;
        ALPHA_MAX   = 4.0;
        
        FREQ_C      = 1.0;   % Coupling frequency
    end
    
    properties (Constant)
        NAME = 'Solvated Periodic Grid';
        ABBR = 'SL';
        DIM  = 16;
        
        % Visualization Limits
        X_PERI_LIM = [-pi, pi];
        X_FREE_LIM_COMP  = [-6.0, 6.0];
        X_FREE_LIM_PLOT  = [-4.0, 4.0];
    end
    
    methods
        function obj = Solvated_16D(varargin)
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
            addParameter(p, 'ALPHA_MIN', obj.ALPHA_MIN);
            addParameter(p, 'ALPHA_MAX', obj.ALPHA_MAX);
            
            parse(p, varargin{:});
            fields = fieldnames(p.Results);
            for i = 1:length(fields), obj.(fields{i}) = p.Results.(fields{i}); end
            
            % Initialize Variable ALPHA_C
            % Linearly spaced from ALPHA_MIN to ALPHA_MAX across the 14 bath dimensions
            obj.ALPHA_C = linspace(obj.ALPHA_MIN, obj.ALPHA_MAX, obj.DIM - 2); 
            
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
        end
        
        function obj = init_target_parameters(obj)
            % Deterministic
        end
        
        function u = compute_potential(obj, x)
            u = obj.compute_potential_mixed(x, 1.0);
        end
        
        function u_mix = compute_potential_mixed(obj, x, lambda)
            % Input: x [N, 16]
            
            % 1. Base Potential (Topologically Mixed)
            % Dims 1-2: Uniform on Torus => Potential = 0 (Flat)
            % Dims 3-16: Gaussian Bath => Potential = 0.5 * z^2
            
            % Gaussian contribution from Bath Dims only
            x_bath = x(:, 3:end);
            z_bath = (x_bath - obj.MEAN) ./ obj.SIGMA;
            u_base = 0.5 * sum(z_bath.^2, 2);
            
            % 2. Target Potential (Solvated)
            x1 = x(:,1); x2 = x(:,2);
            % x_bath already extracted
            
            % Part A: Periodic Grid
            % Minima at cos(4x) = -1 => 4x = pi, 3pi...
            % We want deep wells.
            u_grid = obj.A_GRID .* ( cos(obj.FREQ_GRID .* x1) + cos(obj.FREQ_GRID .* x2) );
            
            % Part B: Bath Coupling
            % Equilibrium position for bath depends on x1, x2 (and now k)
            % shift_k = alpha_k * sin(w*x1) * sin(w*x2)
            % ALPHA_C is [1, 14], sin(...) is [N, 1]
            shift_val = obj.ALPHA_C .* (sin(obj.FREQ_C .* x1) .* sin(obj.FREQ_C .* x2));
            % shift_val is [N, 14]. Broadcast works.
            
            diff = x_bath - shift_val; 
            
            % Constant Stiffness: 0.5 * K * sum(diff^2)
            temp = sum(diff.^2, 2);
            u_bath = 0.5 * obj.K_SPRING .* temp;
            
            u_target = u_grid + u_bath;
            
            % 3. Mix
            u_mix = (1.0 - lambda) .* u_base + lambda .* u_target;
        end
        
        function grad_mix = compute_gradient_mixed(obj, x, lambda)
            % 1. Base Gradient
            grad_base = zeros(size(x));
            
            % Dims 1-2: Uniform => Gradient = 0
            % Dims 3-16: Gaussian => Gradient = z
            grad_base(:, 3:end) = (x(:, 3:end) - obj.MEAN) ./ (obj.SIGMA^2);
            
            % 2. Target Gradient
            grad_target = zeros(size(x));
            x1 = x(:,1); x2 = x(:,2);
            x_bath = x(:, 3:end);
            
            % Precompute common terms
            S1 = sin(obj.FREQ_C .* x1); C1 = cos(obj.FREQ_C .* x1);
            S2 = sin(obj.FREQ_C .* x2); C2 = cos(obj.FREQ_C .* x2);
            
            % shift_k = alpha_k * S1 * S2
            shift = obj.ALPHA_C .* (S1 .* S2); % [N, 14]
            
            % Diff [N, 14]
            diff = x_bath - shift;
            
            % A. Bath Gradients (x3...x16)
            % dU/dx_k = K * (x_k - shift_k)
            grad_target(:, 3:end) = obj.K_SPRING .* diff;
            
            % B. Coupling Gradient on x1, x2 (Chain rule)
            % dU_bath/dx1 = sum_k [ dU/ds_k * ds_k/dx1 ]
            % dU/ds_k = -K * (x_k - s_k) = -K * diff_k
            % ds_k/dx1 = alpha_k * w * cos(w*x1) * sin(w*x2)
            
            common_grad_bath = -obj.K_SPRING .* diff; % [N, 14]
            
            d_shift_d_x1 = obj.ALPHA_C .* (obj.FREQ_C .* C1 .* S2); % [N, 14]
            d_shift_d_x2 = obj.ALPHA_C .* (obj.FREQ_C .* S1 .* C2); % [N, 14]
            
            % Element-wise product [N, 14] then sum over k
            grad_c_x1 = sum(common_grad_bath .* d_shift_d_x1, 2); % [N, 1]
            grad_c_x2 = sum(common_grad_bath .* d_shift_d_x2, 2); % [N, 1]
            
            % C. Grid Gradients
            % U_grid = A(cos4x1 + cos4x2)
            % dU/dx1 = -4A sin(4x1)
            grad_g_x1 = -obj.FREQ_GRID .* obj.A_GRID .* sin(obj.FREQ_GRID .* x1);
            grad_g_x2 = -obj.FREQ_GRID .* obj.A_GRID .* sin(obj.FREQ_GRID .* x2);
            
            grad_target(:,1) = grad_g_x1 + grad_c_x1;
            grad_target(:,2) = grad_g_x2 + grad_c_x2;
            
            % 3. Mix
            grad_mix = (1.0 - lambda) .* grad_base + lambda .* grad_target;
        end
        
        function x_next = integrator(obj, x)
             % Overdamped Langevin Dynamics
            lambdas = obj.BETA_LIST; 
            dt      = obj.DT; % Use stiffer DT
            
            grad = obj.compute_gradient_mixed(x, lambdas);
            
            raw_drift = -grad * dt;
            % Basic Euler-Maruyama for now, Taming if needed for stiff bath
            % taming_factor = 1 + dt * abs(grad); 
            % tamed_drift = raw_drift ./ taming_factor;
            
            noise = randn(size(x));
            sigma = sqrt(2 * dt); 
            diffusion = sigma .* noise;
            
            x_next = x + raw_drift + diffusion;
            
            % 4. Enforce Periodicity on x1, x2 (Wrap to [-pi, pi])
            % This is crucial for maintaining the Uniform distribution on T^2
            % when the potential is flat (U=0).
            x_next(:, 1) = mod(x_next(:, 1) + pi, 2*pi) - pi;
            x_next(:, 2) = mod(x_next(:, 2) + pi, 2*pi) - pi;
        end
        
        function grad = compute_gradient(obj, x)
            grad = obj.compute_gradient_mixed(x, 1.0);
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d (2 Periodic + 14 Bath)\n', obj.DIM);
            fprintf('Params: Grid %dx%d, K_bath=%.1f, Alpha_range=[%.1f, %.1f]\n', ...
                obj.FREQ_GRID, obj.FREQ_GRID, obj.K_SPRING, obj.ALPHA_MIN, obj.ALPHA_MAX);
            fprintf('----------------------------------\n');
        end
    end
end