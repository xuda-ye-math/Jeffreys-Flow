classdef Annulus
    % ANNULUS Potential Energy System Class
    %
    % Description:
    %   Defines the potential energy U(x), gradient, and Langevin dynamics
    %   integrator for the Annulus potential.
    %   U(x) = 8 * ( |x|^6 - 8|x|^4 + 16|x|^2 + 1 )^(1/3)
    %
    %   Note: This class uses standard Euler-Maruyama integration (no taming).

    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 20000;    % Number of target samples
        MEAN        = 0;        % Mean of base distribution
        SIGMA       = 1;        % Std of base distribution
        PLOT_CHOICE = 3;        % Choice for plotting distribution
        
        % ---- 2. PT / Annealing Schedule ----
        BETA_1      = 0.2;      % Starting inverse temperature
        PT_M        = 6;        % Number of replicas
        BETA_LIST               % [M, 1] Array of inverse temperatures
        
        % ---- 3. Dynamics Parameters ----
        DT          = 1e-2;     % Time step
        COR_TIME    = 1;      % Correlation time
        CONST_SWAP  = 0.025;    % Swap frequency (fraction of COR_TIME)
        WARM_TIME   = 200;      % Physical warm-up time
        
        % ---- 4. Derived Parameters ----
        SWAP_INTERVAL_STEPS     % Steps between swaps
    end

    properties (Constant)
        NAME = 'Annulus';
        ABBR = 'AN';
        DIM  = 2;
        
        % Limits for Plotting/Computing
        X_LIM_COMPUTE = [-4, 4];
        Y_LIM_COMPUTE = [-4, 4];
        X_LIM_PLOT    = [-3, 3];
        Y_LIM_PLOT    = [-3, 3];
    end

    methods
        function obj = Annulus(varargin)
            % Constructor: Configures parameters via Name-Value pairs
            p = inputParser;
            addParameter(p, 'MU_SIZE', obj.MU_SIZE);
            addParameter(p, 'MEAN', obj.MEAN);
            addParameter(p, 'SIGMA', obj.SIGMA);
            addParameter(p, 'BETA_1', obj.BETA_1);
            addParameter(p, 'PT_M', obj.PT_M);
            addParameter(p, 'DT', obj.DT);
            addParameter(p, 'COR_TIME', obj.COR_TIME);
            addParameter(p, 'CONST_SWAP', obj.CONST_SWAP);
            addParameter(p, 'WARM_TIME', obj.WARM_TIME);
            
            parse(p, varargin{:});
            
            % Update properties
            fields = fieldnames(p.Results);
            for i = 1:length(fields)
                obj.(fields{i}) = p.Results.(fields{i});
            end

            % Construct Temperature Ladder (Geometric Spacing)
            if obj.PT_M > 1
                obj.BETA_LIST = logspace(log10(obj.BETA_1), 0, obj.PT_M)';
            else
                obj.BETA_LIST = 1;
            end
            
            % Calculate Swap Interval
            % Ensure at least 1 step
            swap_time = obj.CONST_SWAP * obj.COR_TIME;
            obj.SWAP_INTERVAL_STEPS = max(1, round(swap_time / obj.DT));
        end

        function u = compute_potential(~, x)
            % Compute Potential Energy U(x)
            % U(x) = 10 * (r^6 - 8r^4 + 16r^2 + 1)^(1/3)
            % where r^2 = |x|^2
            % Input: x [M, 2]
            % Output: u [M, 1]
            
            % Calculate r^2
            r2 = sum(x.^2, 2);
            
            % Inner term: r^6 - 8r^4 + 16r^2 + 1
            % (r^2 - 4)^2 * r^2 + 1 >= 1, so safe for real root
            inner_term = r2.^3 - 8.*r2.^2 + 16.*r2 + 1;
            
            u = 10 .* inner_term.^(1/3);
        end

        function grad = compute_gradient(~, x)
            % Compute Gradient nabla U(x)
            % Let f(r^2) = (r^2)^3 - 8(r^2)^2 + 16(r^2) + 1
            % U = 10 * f(r^2)^(1/3)
            % dU/dx = 10 * (1/3) * f^(-2/3) * f' * (2x)
            
            % Input: x [M, 2]
            % Output: grad [M, 2]
            
            r2 = sum(x.^2, 2);
            
            % f(r^2)
            f_val = r2.^3 - 8.*r2.^2 + 16.*r2 + 1;
            
            % df/d(r^2) = 3(r^2)^2 - 16(r^2) + 16
            df_dr2 = 3.*r2.^2 - 16.*r2 + 16;
            
            % Scalar coefficient for gradient
            % coeff = (20/3) * f^(-2/3) * df_dr2
            coeff = (20/3) .* (f_val.^(-2/3)) .* df_dr2;
            
            % Apply to x direction
            grad = coeff .* x;
        end
        
        function x_next = integrator(obj, x)
            % Standard Euler-Maruyama Integrator (No Taming)
            % Uses internal properties (DT, BETA_LIST)
            % Input: x [M, 2]
            
            % 1. Parameters
            betas = obj.BETA_LIST;
            dt    = obj.DT;
            
            % 2. Gradient
            grad = obj.compute_gradient(x);
            
            % 3. Standard Drift
            % x_{k+1} = x_k - grad * dt + diffusion
            drift = -grad * dt;
            
            % 4. Diffusion
            noise = randn(size(x));
            sigma = sqrt(2 * dt ./ betas);
            diffusion = sigma .* noise;
            
            % 5. Update
            x_next = x + drift + diffusion;
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) Configuration ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d\n', obj.DIM);
            fprintf('Replicas (PT_M): %d, Beta_1: %.3f\n', obj.PT_M, obj.BETA_1);
            fprintf('Steps per Swap: %d\n', obj.SWAP_INTERVAL_STEPS);
            fprintf('----------------------------------\n');
        end
    end
end