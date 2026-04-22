%% 1D-Euler equation solver.
% case: Sod shock tube
% time advancing: 3-order Runge-Kutta
% flux vector splitting: Steger-Warming
% flux reconstruction: 5-order WENO-JS

clear; clc;

%% Parameters config.
gamma = 5/3;                        % heat ratio
Nx = 100;                           % grids
xmin = 0; xmax = 1;                 % space domain
dx = (xmax - xmin) / Nx;            % grid size
x = xmin + dx/2 : dx : xmax - dx/2; % cell centers

T_final = 0.14;        % final time
CFL = 0.01;            % CFL number

%% Initialization.
% x < 0.5: (rho, u, p) = (1, 0, 1)
% x > 0.5: (rho, u, p) = (0.125, 0, 0.1)

rho = zeros(1, Nx);
u = zeros(1, Nx);
p = zeros(1, Nx);

for i = 1:Nx
    if x(i) < 0.5
        rho(i) = 1.0;
        u(i) = 0.0;
        p(i) = 1.0;
    else
        rho(i) = 0.125;
        u(i) = 0.0;
        p(i) = 0.1;
    end
end

% Transform primitive variables into conserved variables.
[rho, rhou, E] = primitive2conserved(rho, u, p, gamma);

% Ghost cells for boundary.
Nghost = 3;
rho = extend_ghost(rho, Nghost);
rhou = extend_ghost(rhou, Nghost);
E = extend_ghost(E, Nghost);

%% Time advancing.
t = 0;
U = [rho; rhou; E];  % 3 x (Nx + 2*Nghost)

while t < T_final
    % Calculate time step according to CFL number.
    [rho_c, u_c, p_c] = conserved2primitive(U(1,:), U(2,:), U(3,:), gamma);
    c = sqrt(gamma * p_c ./ rho_c);  % 声速
    lambda_max = max(abs(u_c) + c);
    dt = CFL * dx / lambda_max;
    if t + dt > T_final
        dt = T_final - t;
    end
    
    % RK3 Step 1
    L = compute_rhs_fvs(U, dx, gamma, Nghost);
    U1 = U + dt * L;
    
    % RK3 Step 2
    L = compute_rhs_fvs(U1, dx, gamma, Nghost);
    U2 = 0.75*U + 0.25*U1 + 0.25*dt*L;
    
    % RK3 Step 3
    L = compute_rhs_fvs(U2, dx, gamma, Nghost);
    U = 1/3*U + 2/3*U2 + 2/3*dt*L;
    
    t = t + dt;
    fprintf('t = %.4f, dt = %.6f\n', t, dt);
end

%% Results.
[rho_final_weno5, u_final_weno5, p_final_weno5] = conserved2primitive(...
    U(1, Nghost+1:Nghost+Nx), U(2, Nghost+1:Nghost+Nx), U(3, Nghost+1:Nghost+Nx), gamma);

%% Plotting ρ.
subplot(1,3,1);
plot(x, rho_final_weno5, 'b-', 'LineWidth', 1.5, 'DisplayName', 'WENO5');
hold on;
title('Density \rho');
xlabel('x'); ylabel('\rho');
legend('show')
grid on; xlim([0 1]);
%% Plotting u.
subplot(1,3,2);
plot(x, u_final_weno5, 'b-', 'LineWidth', 1.5, 'DisplayName', 'WENO5');
title('Velocity u');
xlabel('x'); ylabel('u');
legend('show')
grid on; xlim([0 1]);
%% Plotting p.
subplot(1,3,3);
plot(x, p_final_weno5, 'b-', 'LineWidth', 1.5, 'DisplayName', 'WENO5');
title('Pressure p');
xlabel('x'); ylabel('p');
legend('show')
grid on; xlim([0 1]);

sgtitle('Sod Shock Tube - 5th-order WENO + FVS (Steger-Warming), t = 0.2');

%% Save results.
save('sod_weno5_fvs_results.mat', 'x', 'rho_final_weno5', 'u_final_weno5', 'p_final_weno5');

%% Utilities.

function [rho, rhou, E] = primitive2conserved(rho, u, p, gamma)
    % Primitive => conserved
    rhou = rho .* u;
    E = p / (gamma - 1) + 0.5 * rho .* u.^2;
end

function [rho, u, p] = conserved2primitive(rho, rhou, E, gamma)
    % Conserved => primitive
    u = rhou ./ rho;
    p = (gamma - 1) * (E - 0.5 * rhou.^2 ./ rho);
end

function U_ext = extend_ghost(U, Nghost)
    % Constant boundary.
    N = length(U);
    U_ext = zeros(1, N + 2*Nghost);
    U_ext(Nghost+1:Nghost+N) = U;
    % left
    for i = 1:Nghost
        U_ext(Nghost+1-i) = U(1);
    end
    % right
    for i = 1:Nghost
        U_ext(Nghost+N+i) = U(end);
    end
end

function L = compute_rhs_fvs(U, dx, gamma, Nghost)
    % Calculate right hand side using FVS: -dF/dx
    % Steger-Warming spliting: F = F^+ + F^-
    
    [~, Ntot] = size(U);
    Nx = Ntot - 2*Nghost;
    
    % Primitives
    [rho, u, p] = conserved2primitive(U(1,:), U(2,:), U(3,:), gamma);
    c = sqrt(gamma * p ./ rho);
    
    % FVS: F^+ + F^- (Steger-Warming)
    [Fp, Fm] = steger_warming_split(U, gamma, rho, u, c);
    
    F_num = zeros(3, Ntot-1);  % numerical flux at interface i+1/2
    
    % Calculate flux at every interface.
    for i = Nghost : Nghost+Nx
        % F^+ reconstruction at interface i+1/2
        Fp_stencil = Fp(:, i-2:i+2);
        Fp_left = zeros(3,1);
        for k = 1:3
            Fp_left(k) = weno5_reconstruct_positive(Fp_stencil(k, :));
        end
        
        % F^- reconstruction at interface i+1/2
        Fm_stencil = Fm(:, i-1:i+3);
        Fm_right = zeros(3,1);
        for k = 1:3
            Fm_right(k) = weno5_reconstruct_negtive(Fm_stencil(k, :));
        end
        
        % Numerical flux at interface: F = F^ + F^-
        F_num(:, i) = Fp_left + Fm_right;
    end
    
    % L = -dF/dx
    L = zeros(3, Ntot);
    L(:, Nghost+1:Nghost+Nx) = -(F_num(:, Nghost+1:Nghost+Nx) - F_num(:, Nghost:Nghost+Nx-1)) / dx;
end

function [Fp, Fm] = steger_warming_split(U, gamma, rho, u, c)
    % Steger-Warming splitting
    % F = F^+ + F^-
    
    [~, N] = size(U);
    Fp = zeros(3, N);
    Fm = zeros(3, N);
    
    for i = 1:N
        
        % Eigen value: lambda1 = u, lambda2 = u-c, lambda3 = u+c
        lambda1 = u(i);
        lambda2 = u(i) - c(i);
        lambda3 = u(i) + c(i);
        
        % Eigen value splitting.
        epsilon = 1e-6;
        split = @(lambda) [
            0.5*(lambda+sqrt(lambda^2+epsilon^2));
            0.5*(lambda-sqrt(lambda^2+epsilon^2))
        ];
        lam1s = split(lambda1);
        lam2s = split(lambda2);
        lam3s = split(lambda3);
        [lambda1p, lambda1m] = deal(lam1s(1), lam1s(2));
        [lambda2p, lambda2m] = deal(lam2s(1), lam2s(2));
        [lambda3p, lambda3m] = deal(lam3s(1), lam3s(2));

        % Flux splitting.
        rho_i = rho(i);
        u_i = u(i);
        c_i = c(i);

        flux = @(la1,la2,la3,rho,u,c,gamma) 0.5*rho/gamma*[
            2*(gamma-1)*la1 + la2 + la3;
            2*(gamma-1)*la1*u + la2*(u-c) + la3*(u+c);
            (gamma-1)*la1*u*u + 0.5*la2*(u-c)^2 + 0.5*la3*(u+c)^2 + (3-gamma)*(la2+la3)*c*c/(2*(gamma-1))
        ];
        
        Fp(:,i) = flux(lambda1p, lambda2p, lambda3p, rho_i, u_i, c_i, gamma);
        Fm(:,i) = flux(lambda1m, lambda2m, lambda3m, rho_i, u_i, c_i, gamma);
    end
end

function w = weno5_reconstruct_positive(v)
    % 5-order WENO-JS reconstruction: v_{i+1/2}^+
    % input: v contains [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
    
    eps = 1e-6;
    
    % 3-order sub-construction.
    v0 = (1/3)*v(1) - (7/6)*v(2) + (11/6)*v(3);
    beta0 = (13/12)*(v(1) - 2*v(2) + v(3))^2 + (1/4)*(v(1) - 4*v(2) + 3*v(3))^2;
    
    v1 = -(1/6)*v(2) + (5/6)*v(3) + (1/3)*v(4);
    beta1 = (13/12)*(v(2) - 2*v(3) + v(4))^2 + (1/4)*(v(2) - v(4))^2;
    
    v2 = (1/3)*v(3) + (5/6)*v(4) - (1/6)*v(5);
    beta2 = (13/12)*(v(3) - 2*v(4) + v(5))^2 + (1/4)*(3*v(3) - 4*v(4) + v(5))^2;
    
    % Ideal weights.
    d0 = 0.1; d1 = 0.6; d2 = 0.3;
    
    % WENO-JS weights
    alpha0 = d0 / (eps + beta0)^2;
    alpha1 = d1 / (eps + beta1)^2;
    alpha2 = d2 / (eps + beta2)^2;
    
    % Normalized WENO-JS weights
    w_sum = alpha0 + alpha1 + alpha2;
    w0 = alpha0 / w_sum;
    w1 = alpha1 / w_sum;
    w2 = alpha2 / w_sum;
    
    w = w0*v0 + w1*v1 + w2*v2;
end

function w = weno5_reconstruct_negtive(v)
    % 5-order WENO-JS reconstruction: v_{i+1/2}^-
    % input: v contains [v_{i-1}, v_{i}, v_{i+1}, v_{i+2}, v_{i+3}]
    
    eps = 1e-6;
    
    % 3-order sub-construction.
    v0 = (1/3)*v(5) - (7/6)*v(4) + (11/6)*v(3);
    beta0 = (13/12)*(v(5) - 2*v(4) + v(3))^2 + (1/4)*(v(5) - 4*v(4) + 3*v(3))^2;
    
    v1 = -(1/6)*v(4) + (5/6)*v(3) + (1/3)*v(2);
    beta1 = (13/12)*(v(4) - 2*v(3) + v(2))^2 + (1/4)*(v(4) - v(2))^2;
    
    v2 = (1/3)*v(3) + (5/6)*v(2) - (1/6)*v(1);
    beta2 = (13/12)*(v(3) - 2*v(2) + v(1))^2 + (1/4)*(3*v(3) - 4*v(2) + v(1))^2;
    
    % Ideal weights.
    d0 = 0.1; d1 = 0.6; d2 = 0.3;
    
    % WENO-JS weights
    alpha0 = d0 / (eps + beta0)^2;
    alpha1 = d1 / (eps + beta1)^2;
    alpha2 = d2 / (eps + beta2)^2;
    
    % Normalized WENO-JS weights
    w_sum = alpha0 + alpha1 + alpha2;
    w0 = alpha0 / w_sum;
    w1 = alpha1 / w_sum;
    w2 = alpha2 / w_sum;
    
    w = w0*v0 + w1*v1 + w2*v2;
end
