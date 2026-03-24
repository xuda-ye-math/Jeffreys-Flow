classdef Periodic_Well
    % PERIODIC_WELL Potential Energy System Class
    %
    % Description:
    %   Defines the potential energy U(x), gradient, and Langevin dynamics
    %   integrator.
    %   
    %   Potential Function (Corrected for Real-valued computation):
    %     U(x,y) = 4 * sin(2*x) * sign(sin(2*y)) * |sin(2*y)|^(7/5)
    %
    %   Gradient:
    %     Derived consistently to ensure real values.
    %
    %   Note: This potential has a "gap" (energy barrier/range) of approx 8.

    properties
        % ---- 1. Configurable Parameters ----
        MU_SIZE     = 20000;    % Number of target samples
        PLOT_CHOICE = 2;        % Choice for plotting distribution
        
        % ---- 2. PT / Annealing Schedule ----
        BETA_1      = 0.2;      % Starting inverse temperature
        PT_M        = 8;       % Number of replicas (increased for better swapping)
        BETA_LIST               % [M, 1] Array of inverse temperatures
        
        % ---- 3. Dynamics Parameters ----
        DT          = 2e-2;     % Time step
        COR_TIME    = 2;        % Correlation time
        CONST_SWAP  = 0.025;    % Swap frequency (fraction of COR_TIME)
        WARM_TIME   = 100;      % Physical warm-up time
        
        % ---- 4. Derived Parameters ----
        SWAP_INTERVAL_STEPS     % Steps between swaps
    end

    properties (Constant)
        NAME = 'Periodic Well';
        ABBR = 'PW';
        DIM  = 2;
        
        % Limits for Plotting/Computing
        X_LIM_PLOT    = [-pi, pi];
        Y_LIM_PLOT    = [-pi, pi];
    end

    methods
        function obj = Periodic_Well(varargin)
            % Constructor: Configures parameters via Name-Value pairs
            p = inputParser;
            addParameter(p, 'MU_SIZE', obj.MU_SIZE);
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
            % Formula: U(x,y) = 4 * sin(2*x) * (sin(2*y))^(7/5)
            % FIX: Use sign() * abs()^p to avoid complex numbers in MATLAB
            % for negative bases.
            
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            sin_2x2 = sin(2.*x2);
            
            % Apply fractional power safely
            term_y = sign(sin_2x2) .* abs(sin_2x2).^(7/5);
            
            u = 4 .* sin(2.*x1) .* term_y;
        end

        function grad = compute_gradient(~, x)
            % Compute Gradient nabla U(x)
            %
            % d/dy ( sign(u)*|u|^p ) = p * |u|^(p-1) * u'
            % Here p = 1.4, so derivative is 1.4 * |u|^0.4 * u'
            % This is always real-valued.
            
            x1 = x(:, 1);
            x2 = x(:, 2);
            
            sin_2x2 = sin(2.*x2);
            cos_2x2 = cos(2.*x2);
            
            % Partial wrt x1
            % d/dx (4*sin(2x)) = 8*cos(2x)
            % Term Y remains same
            term_y = sign(sin_2x2) .* abs(sin_2x2).^(7/5);
            grad_x1 = 8 .* cos(2.*x1) .* term_y;
            
            % Partial wrt x2
            % Chain rule: 4*sin(2x) * [ 1.4 * |sin(2y)|^0.4 * 2*cos(2y) ]
            % 4 * 1.4 * 2 = 11.2
            grad_x2 = 11.2 .* sin(2.*x1) .* abs(sin_2x2).^(0.4) .* cos_2x2;
            
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
            
            % 6. Periodic Boundary Condition (Wrap to [-pi, pi])
            % This prevents particles from drifting into effectively distinct 
            % but mathematically equivalent domains, ensuring correct sampling statistics.
            x_next = mod(x_next + pi, 2*pi) - pi;
        end
        
        function disp_info(obj)
            fprintf('--- %s (%s) Configuration ---\n', obj.NAME, obj.ABBR);
            fprintf('Dimension: %d\n', obj.DIM);
            fprintf('Replicas (PT_M): %d, Beta_1: %.3f\n', obj.PT_M, obj.BETA_1);
            fprintf('Steps per Swap: %d\n', obj.SWAP_INTERVAL_STEPS);
            fprintf('Potential: Real-valued fractional power & PBC enforced.\n');
            fprintf('----------------------------------\n');
        end
    end
end