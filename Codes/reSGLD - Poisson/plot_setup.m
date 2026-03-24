% plot_setup.m
% Script to visualize the 2D Screened Poisson Source and Solution
% Plots a 1x2 figure: left is the actual source field f(x), right is the solution u(x)

clear; clc; close all;

fprintf('--- Plotting Poisson 2D System Setup ---\n');

% 1. Load Parameters
data_dir = './data';
fig_dir = './figures';
para_file = fullfile(data_dir, 'para.mat');

if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(para_file, 'file')
    error('Run parameters.m first to generate para.mat');
end
load(para_file, 'para');

RESOLUTION_DPI = 600;

% Initialize the solver to access the grid and logic
solver = Poisson_2D(para.N_cells, para.alpha, para.gamma, para.c);

% Define True Source Parameters
theta_true = para.theta_true;
theta_true_x = theta_true(1:2:end);
theta_true_y = theta_true(2:2:end);
TRUE_COLOR = [0.1, 0.8, 0.3];

% 2. Calculate the True Source Field Map f(x)
X = solver.X;
Y = solver.Y;
f_mat = zeros(size(X));
num_sources = length(theta_true_x);
c = para.c;
gamma = para.gamma; % Source width

for i = 1:num_sources
    source_x = theta_true_x(i);
    source_y = theta_true_y(i);
    % Gaussian source formulation based on Poisson_2D solver logic (Eq: exp(-r^2 / (2 * gamma^2)))
    f_mat = f_mat + c * exp(-((X - source_x).^2 + (Y - source_y).^2) / (2 * gamma^2));
end

% 3. Calculate the True Solution Field Map u(x)
u_mat = solver.solve_forward_mat(theta_true);

% 4. Visualization Layout Setup (Explicit Parameters)
% -------------------------------------------------------------------------
% Adjust these values to tune the gaps and margins (0.0 to 1.0)
marg_l = 0.05;  % Left margin
marg_r = 0.05;  % Right margin
marg_t = 0.25;  % Top margin
marg_b = 0.12;  % Bottom margin
gap_w  = 0.001;  % <--- Gap between the two columns (TUNE THIS parameter)

% Calculate explicit subplot width and height
plot_w = (1 - marg_l - marg_r - gap_w) / 2;
plot_h = 1 - marg_t - marg_b;

fig_w = 900;
fig_h = 500;
f = figure('Name', '2D Screened Equation Setup', 'Color', 'w', 'Position', [100, 100, fig_w, fig_h]);

% --- Left Subplot (Source Field) ---
ax1 = axes('Position', [marg_l, marg_b, plot_w, plot_h]);
contourf(ax1, X, Y, f_mat, 50, 'LineStyle', 'none');
colormap(ax1, parula);
cb1 = colorbar(ax1);
title(ax1, 'Source Field', 'FontWeight', 'bold', 'FontSize', 12);
axis(ax1, 'square');
hold(ax1, 'on');

% Overlay Source Positions using 4 Green Pentagrams
scatter(ax1, theta_true_x, theta_true_y, 120, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', TRUE_COLOR, 'LineWidth', 1.5, 'DisplayName', 'Source Locations');


% --- Right Subplot (Solution Field) ---
ax2 = axes('Position', [marg_l + plot_w + gap_w, marg_b, plot_w, plot_h]);
contourf(ax2, X, Y, u_mat, 50, 'LineStyle', 'none');
colormap(ax2, parula);
cb2 = colorbar(ax2);
title(ax2, 'Numerical Solution', 'FontWeight', 'bold', 'FontSize', 12);
axis(ax2, 'square');
hold(ax2, 'on');

% Map observation grid exactly matching parameters.m (0.1 : 0.1 : 0.9)
[obs_X, obs_Y] = meshgrid(0.1:0.1:0.9, 0.1:0.1:0.9);

% Overlay Sensor Network Locations
scatter(ax2, obs_X(:), obs_Y(:), 20, 's', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'DisplayName', 'Sensors');

% Overlay Source Positions using 4 Green Pentagrams
scatter(ax2, theta_true_x, theta_true_y, 120, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', TRUE_COLOR, 'LineWidth', 1.5, 'DisplayName', 'Source Locations');

% Link Axes Limits
linkaxes([ax1, ax2], 'xy');

% Global Title
annotation('textbox', [0, 0.85, 1, 0.05], 'String', '2D Screened Poisson System Setup', ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
           'FontSize', 16, 'FontWeight', 'bold');

% 5. Save Figure
save_name = 'P_plot_Poisson_Setup.pdf';
save_path = fullfile(fig_dir, save_name);
fprintf('Saving figure to %s...\n', save_path);
exportgraphics(f, save_path, 'Resolution', RESOLUTION_DPI);

fprintf('Plot generation complete.\n');
