% Poisson_2D.m
% 2D Screened Poisson Solver Class
%
% This class provides a forward solver for the 2D screened Poisson equation:
%     - \Delta u + \alpha u = f
% with pure Neumann boundary conditions.
%
% Spatial domain: [0, 1] x [0, 1] defined on a square grid.
% Discretization: 5-point central finite difference.

classdef Poisson_2D
    properties
        % Grid parameters
        N_cells % Number of cells in both x and y directions
        h       % Grid spacing in both x and y directions
        N       % Total number of grid points ( (N_cells+1)^2 )
        
        % Physical parameters
        alpha   % Screening coefficient
        gamma   % Gaussian source width (standard deviation)
        c       % Uniform source intensity
        
        % Domain coordinates
        X       % 2D meshgrid X coordinates
        Y       % 2D meshgrid Y coordinates
        grid_vec % 1D grid points in x and y
        
        % Discrete Operators
        A       % Sparse system matrix: -Laplacian + alpha*I
    end
    
    methods
        function obj = Poisson_2D(N_cells, alpha, gamma, c)
            % Constructor: Initialize the grid and operators
            % No default parameters allowed.
            
            obj.N_cells = N_cells;
            obj.alpha = alpha;
            obj.gamma = gamma;
            obj.c = c;
            
            % Grid configuration
            obj.h = 1.0 / N_cells;
            
            obj.grid_vec = linspace(0, 1, N_cells + 1)';
            [obj.X, obj.Y] = meshgrid(obj.grid_vec, obj.grid_vec);
            
            obj.N = (N_cells + 1)^2;
            
            % Build the discrete system matrix
            obj = obj.build_matrix();
        end
        
        function obj = build_matrix(obj)
            % Assembles the sparse finite-difference matrix A for the 
            % screened Poisson equation: -Laplacian + alpha*I
            % using 5-point stencil and pure Neumann boundary conditions.
            %
            % Node ordering: column-major (standard MATLAB)
            % Index mapping: k = i + (j-1)*n_pts, where i, j in [1, n_pts]
            
            n_pts = obj.N_cells + 1;
            N_total = n_pts^2;
            
            h2 = obj.h^2;
            
            % Pre-allocate sparse matrix triplets
            % Max 5 entries per row
            max_entries = N_total * 5;
            I = zeros(max_entries, 1);
            J = zeros(max_entries, 1);
            V = zeros(max_entries, 1);
            
            idx = 1;
            for j = 1:n_pts
                for i = 1:n_pts
                    k = i + (j - 1) * n_pts;
                    
                    % Central coefficient
                    coeff_center = 4 / h2 + obj.alpha;
                    
                    % Neighbors
                    c_west  = -1 / h2;
                    c_east  = -1 / h2;
                    c_south = -1 / h2;
                    c_north = -1 / h2;
                    
                    % Apply Neumann Boundary Conditions (Virtual point reflection)
                    if i == 1
                        coeff_center = coeff_center + c_west;
                        c_west = 0;
                    end
                    if i == n_pts
                        coeff_center = coeff_center + c_east;
                        c_east = 0;
                    end
                    if j == 1
                        coeff_center = coeff_center + c_south;
                        c_south = 0;
                    end
                    if j == n_pts
                        coeff_center = coeff_center + c_north;
                        c_north = 0;
                    end
                    
                    % Fill diagonal
                    I(idx) = k; J(idx) = k; V(idx) = coeff_center; idx = idx + 1;
                    
                    % Off-diagonals
                    if c_west ~= 0
                        I(idx) = k; J(idx) = k - 1; V(idx) = c_west; idx = idx + 1;
                    end
                    if c_east ~= 0
                        I(idx) = k; J(idx) = k + 1; V(idx) = c_east; idx = idx + 1;
                    end
                    if c_south ~= 0
                        I(idx) = k; J(idx) = k - n_pts; V(idx) = c_south; idx = idx + 1;
                    end
                    if c_north ~= 0
                        I(idx) = k; J(idx) = k + n_pts; V(idx) = c_north; idx = idx + 1;
                    end
                end
            end
            
            I = I(1:idx-1);
            J = J(1:idx-1);
            V = V(1:idx-1);
            
            % Create sparse matrix
            obj.A = sparse(I, J, V, N_total, N_total);
        end
        
        function f = build_source(obj, theta)
            % Constructs the source term right-hand side f(x; theta)
            % theta: [8x1] or [1x8] vector containing [x1, y1, x2, y2, x3, y3, x4, y4]
            % Returns: f as [N x 1] vector
            
            theta = theta(:); % Ensure column vector
            num_sources = length(theta) / 2;
            
            f_mat = zeros(size(obj.X));
            
            for k = 1:num_sources
                mu_x = theta(2*k - 1);
                mu_y = theta(2*k);
                
                gaussian = obj.c * exp(-((obj.X - mu_x).^2 + (obj.Y - mu_y).^2) / (2 * obj.gamma^2));
                f_mat = f_mat + gaussian;
            end
            
            f = f_mat(:); % Flatten to column vector
        end
        
        function u = solve_forward(obj, theta)
            % Solves the forward problem: A * u = f(theta)
            % Returns: u as [N x 1] vector
            
            f = obj.build_source(theta);
            
            % Directly solve using MATLAB's sparse direct solver (mldivide)
            u = obj.A \ f;
        end
        
        function u_mat = solve_forward_mat(obj, theta)
            % Solves the forward problem and returns a 2D matrix
            % Returns: u_mat as [(N_cells+1) x (N_cells+1)] matrix (same shape as X and Y)
            
            u_vec = obj.solve_forward(theta);
            u_mat = reshape(u_vec, size(obj.X));
        end
        
        function grad_theta = compute_gradient_f_theta(obj, theta, lambda)
            % Compute the gradient of the potential with respect to theta 
            % using the adjoint state lambda.
            % Formula: grad_theta = (\partial f / \partial \theta)^T * lambda
            %
            % Inputs:
            %   theta  : [8x1] vector of source coordinates [x1, y1, x2, y2, ...]
            %   lambda : [Nx1] vector of the adjoint state from (A \ v)
            %
            % Output:
            %   grad_theta : [8x1] gradient vector
            
            theta = theta(:);
            num_sources = length(theta) / 2;
            grad_theta = zeros(8, 1);
            
            gamma2 = obj.gamma^2;
            
            % Lambda needs to be the same shape as X and Y to vectorize
            lambda_mat = reshape(lambda, size(obj.X));
            
            for k = 1:num_sources
                mu_x = theta(2*k - 1);
                mu_y = theta(2*k);
                
                % Contribution of this specific source k to the field f
                f_k = obj.c * exp(-((obj.X - mu_x).^2 + (obj.Y - mu_y).^2) / (2 * gamma2));
                
                % df/dx_k = f_k * (x - mu_x) / gamma^2
                df_dx_k = f_k .* (obj.X - mu_x) / gamma2;
                
                % df/dy_k = f_k * (y - mu_y) / gamma^2
                df_dy_k = f_k .* (obj.Y - mu_y) / gamma2;
                
                % Inner product with lambda (sum over the whole grid)
                grad_theta(2*k - 1) = sum(sum(df_dx_k .* lambda_mat));
                grad_theta(2*k)     = sum(sum(df_dy_k .* lambda_mat));
            end
        end
    end
end
