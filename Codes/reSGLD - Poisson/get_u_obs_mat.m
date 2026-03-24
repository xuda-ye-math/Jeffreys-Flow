function [u_obs_mat, obs_indices] = get_u_obs_mat(u_mat, N_cells)
% GET_U_OBS_MAT Extracts the observation values from the full solution matrix.
%
% This function maps the full state variable u on the grid to the sparse 
% 9x9 sensor network defined strictly inside the domain.
%
% Sensors are located at (m * 0.1, n * 0.1) for m, n in {1, 2, ..., 9}.
%
% Inputs:
%   u_mat   - [ (N_cells+1) x (N_cells+1) ] The full state variable matrix
%   N_cells - The number of grid cells in the domain [100 for default setup]
%             A cell spacing of h = 1/N_cells means observation coordinates 
%             are exactly on the grid points when N_cells is a multiple of 10.
%
% Outputs:
%   u_obs_mat   - [9 x 9] matrix containing the extracted observations.
%                 u_obs_mat(i, j) corresponds to observation at x=i*0.1, y=j*0.1.
%   obs_indices - [81 x 1] absolute linear indices in the original u_mat 
%                 corresponding to the sensor locations.

    % The grid spacing is h = 1 / N_cells.
    % To place sensors exactly at 0.1, 0.2, ... 0.9, N_cells must be a multiple of 10.
    if mod(N_cells, 10) ~= 0
        error('N_cells must be a multiple of 10 for exact observation placement (e.g., 40, 100).');
    end
    
    % The grid spacing is h = 1 / N_cells.
    % The point coordinates are x_i = (i-1)*h.
    % So x = 0.1 corresponds to index i where (i-1)*(1/N_cells) = 0.1
    % Therefore i = 1 + N_cells * 0.1
    step_idx = round(N_cells * 0.1);
    
    % The indices for 0.1, 0.2, ... 0.9
    obs_x_idx = 1 + step_idx : step_idx : 1 + 9 * step_idx;
    obs_y_idx = 1 + step_idx : step_idx : 1 + 9 * step_idx;
    
    % MATLAB arrays are (row, col) which corresponds to (y, x) in meshgrid format
    % Recall from Poisson_2D.m: [X, Y] = meshgrid(grid_vec, grid_vec)
    % u_mat has the same shape as X and Y
    % u_mat(row, col) corresponds to y_idx = row, x_idx = col
    
    u_obs_mat = u_mat(obs_y_idx, obs_x_idx);
    
    % Compute the linear indices if requested
    if nargout > 1
        n_pts = N_cells + 1;
        
        % Generate the meshgrid of indices for the observation points
        [Obs_X_idx, Obs_Y_idx] = meshgrid(obs_x_idx, obs_y_idx);
        
        % Compute absolute linear indices using column-major order:
        % linear_idx = row + (col - 1) * max_rows
        obs_indices = Obs_Y_idx(:) + (Obs_X_idx(:) - 1) * n_pts;
    end
end
