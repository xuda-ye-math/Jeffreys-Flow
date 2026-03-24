classdef Multiple_Well
    % MULTIPLE_WELL Potential Energy System Class
    %
    % Description:
    %   Defines the potential energy U(x), gradient, and Langevin dynamics
    %   integrator for the Multiple Well potential.
    %   U(x,y) = 4*sin(1.1*x)*sin(y) + 0.25*x^2 + 0.25*y^2
    %
    %   Integrator: Stochastic Heun (Predictor-Corrector)

    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 20000;    % Number of target samples
        MEAN        = 0;        % Mean of base distribution
        SIGMA       = 2.2;      % Std of base distribution (Wider to cover wells)
        PLOT_CHOICE = 2;        % Choice for plotting distribution
        
        % ---- 2. PT / Annealing Schedule ----
        BETA_1      = 0.1;      % Starting inverse temperature
        PT_M        = 6;        % Number of replicas
        BETA_LIST               % [M, 1] Array of inverse temperatures
        
        % ---- 3. Dynamics Parameters ----
        DT          = 1;     % Time step
        COR_TIME    = 4.0;      % Correlation time
        CONST_SWAP  = 0.025;    % Swap frequency (fraction of COR_TIME)
        WARM_TIME   = 100;      % Physical warm-up time
        
        % ---- 4. Derived Parameters ----
        SWAP_INTERVAL_STEPS     % Steps between swaps
    end

    properties (Constant)
        NAME = 'Multiple Well';
        ABBR = 'MW';
        DIM  = 2;
        
        % Limits for Plotting/Computing
        X_LIM_COMPUTE = [-7, 7];
        Y_LIM_COMPUTE = [-7, 7];
        X_LIM_PLOT    = [-5, 5];
        Y_LIM_PLOT    = [-5, 5];
    end

    methods
        function obj = Multiple_Well(varargin)
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
            % U(x,y) = 4*sin(1.1*x)*sin(y) + 0.25*x^2 + 0.25*y^2
            % Input: x [M, 2]
            % Output: u [M, 1]
            
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            % Term 1: 4 * sin(1.1*x) * sin(y)
            term_sin = 4 .* sin(1.1 .* x1) .* sin(x2);
            
            % Term 2: 0.25 * x^2 + 0.25 * y^2
            term_quad = 0.25 .* x1.^2 + 0.25 .* x2.^2;
            
            u = term_sin + term_quad;
        end

        function grad = compute_gradient(~, x)
            % Compute Gradient nabla U(x)
            % U = 4*sin(1.1x)sin(y) + 0.25x^2 + 0.25y^2
            
            % dU/dx = 4 * 1.1 * cos(1.1x)sin(y) + 0.5x
            %       = 4.4 * cos(1.1x)sin(y) + 0.5x
            
            % dU/dy = 4 * sin(1.1x)cos(y) + 0.5y
            
            % Input: x [M, 2]
            % Output: grad [M, 2]
            
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            % Partial wrt x1
            grad_x1 = 4.4 .* cos(1.1 .* x1) .* sin(x2) + 0.5 .* x1;
            
            % Partial wrt x2
            grad_x2 = 4.0 .* sin(1.1 .* x1) .* cos(x2) + 0.5 .* x2;
            
            grad = [grad_x1, grad_x2];
        end
        
        function x_next = integrator(obj, x)
            % Stochastic Heun Integrator (Predictor-Corrector)
            % For dX = -grad(U)dt + sigma*dW
            %
            % Uses internal properties (DT, BETA_LIST)
            % Input: x [M, 2]
            
            % 1. Parameters
            betas = obj.BETA_LIST;
            dt    = obj.DT;
            
            % 2. Calculate Noise (Shared for Predictor and Corrector)
            % sigma = sqrt(2 * dt / beta)
            noise = randn(size(x));
            sigma = sqrt(2 * dt ./ betas);
            diffusion = sigma .* noise;
            
            % 3. Predictor Step (Euler-Maruyama)
            grad_1 = obj.compute_gradient(x);
            x_tilde = x - grad_1 * dt + diffusion;
            
            % 4. Corrector Step (Trapezoidal Drift)
            grad_2 = obj.compute_gradient(x_tilde);
            drift_corrected = -0.5 * (grad_1 + grad_2) * dt;
            
            % x_{k+1} = x_k + 0.5(a(x_k) + a(x_tilde))dt + b*dW
            x_next = x + drift_corrected + diffusion;
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) Configuration ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d\n', obj.DIM);
            fprintf('Replicas (PT_M): %d, Beta_1: %.3f\n', obj.PT_M, obj.BETA_1);
            fprintf('Steps per Swap: %d\n', obj.SWAP_INTERVAL_STEPS);
            fprintf('Integrator: Stochastic Heun\n');
            fprintf('----------------------------------\n');
        end
    end
end