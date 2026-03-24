classdef Three_Well
    % THREE_WELL Potential Energy System Class
    %
    % Description:
    %   Defines the potential energy U(x), gradient, and Langevin dynamics
    %   integrator. Encapsulates system parameters including the temperature
    %   ladder and time steps.

    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 20000;    % Number of target samples
        MEAN        = 0;        % Mean of base distribution
        SIGMA       = 1;        % Std of base distribution
        PLOT_CHOICE = 2;        % Choice for plotting distribution
        
        % ---- 2. PT / Annealing Schedule ----
        BETA_1      = 0.1;      % Starting inverse temperature
        PT_M        = 5;        % Number of replicas
        BETA_LIST               % [M, 1] Array of inverse temperatures
        
        % ---- 3. Dynamics Parameters ----
        DT          = 1e-1;     % Time step
        COR_TIME    = 1;      % Correlation time
        CONST_SWAP  = 0.025;    % Swap frequency (fraction of COR_TIME)
        WARM_TIME   = 100;      % Physical warm-up time
        
        % ---- 4. Derived Parameters ----
        SWAP_INTERVAL_STEPS     % Steps between swaps
    end

    properties (Constant)
        NAME = 'Three Well';
        ABBR = 'TW';
        DIM  = 2;
        
        % Limits for Plotting/Computing
        X_LIM_COMPUTE = [-4, 4];
        Y_LIM_COMPUTE = [-4, 4];
        X_LIM_PLOT    = [-2, 2];
        Y_LIM_PLOT    = [-2, 2];
    end

    methods
        function obj = Three_Well(varargin)
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
            % Input: x [M, 2]
            % Output: u [M, 1]
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            term1 = (x1.^2 - 1).^2;
            term2 = (x2.^2 - 1).^2;
            term3 = sin(x1 + 2.*x2);
            
            u = 3 * (term1 + term2 + term3);
        end

        function grad = compute_gradient(~, x)
            % Compute Gradient nabla U(x)
            % Input: x [M, 2]
            % Output: grad [M, 2]
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            cos_term = cos(x1 + 2.*x2);
            
            grad_x1 = 3 .* (4 .* x1 .* (x1.^2 - 1) + cos_term);
            grad_x2 = 3 .* (4 .* x2 .* (x2.^2 - 1) + 2 .* cos_term);
            
            grad = [grad_x1, grad_x2];
        end
        
        function x_next = integrator(obj, x)
            % Tamed Euler-Maruyama Integrator
            % Uses internal properties (DT, BETA_LIST)
            % Input: x [M, 2]
            
            % 1. Parameters
            betas = obj.BETA_LIST;
            dt    = obj.DT;
            
            % 2. Gradient
            grad = obj.compute_gradient(x);
            
            % 3. Tamed Drift
            raw_drift = -grad * dt;
            taming_factor = 1 + dt * abs(grad);
            tamed_drift = raw_drift ./ taming_factor;
            
            % 4. Diffusion
            noise = randn(size(x));
            sigma = sqrt(2 * dt ./ betas);
            diffusion = sigma .* noise;
            
            % 5. Update
            x_next = x + tamed_drift + diffusion;
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