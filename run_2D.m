% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

clear; close all; clc; rng default;

%% Set parameters
E = 2;                       % Dimensions of the state space  
dynt = @(t,x) dyn2D(0,x);    % dynamical system to be controlled
dyn = @(x) dynt(0,x);        % time independent handle to dynamics

% Generating Training data
x0tr = [0.3 -0.3; 0 0.1];    % Starting points for training trajectories
dttr = 0.3;                  % Recording step size for taining data (default = 0.3)
Ttr = 3;                     % Simulation time for training per starting point (default = 3)
sn = 1e-1*[1 1]';            % Observation noise (default = 1e-1)
umax = 50;                   % maximum input power (default = 50)
epsbeta = 2;                 % factor by which kc is larger than beta (default = 2)

% GP learning
optGPR = {'FitMethod','exact','ConstantSigma',true,'Sigma',sn,...
    'KernelFunction','ardsquaredexponential' };

% Setup Lyapunov function computation
Ngrid = 1e4;                % Number of gridpoints (default = 1e4)
grid_min = -5.5*[1;1]; grid_max = 5.5*[1;1];
optVvar.Ndgrid = floor(nthroot(Ngrid,E)); %Ngrid = Ndgrid^E;
optVvar.InterpMethod = 'linear';

% Simulation
x0sim = [0;-5];             % Initial point in simulation´
Tsim = 10;                  % Maximum simulation time (default = 10)
dtsim = 0.01;               % Time interval output ode45 (default = 0.01)
epsfd = 1e-4  ;             % finite difference for numeric gradient (default = 1e-4)
r = 0.01;                   % Stopping criteria for simulation  (default = 0.01)
options = odeset('RelTol',1e-3,'AbsTol',1e-6,'Events', @(t,x) isconverged(t,x,r));

% Visualization
Nte = 1e4;                  % Number of points for visualization (default = 1e4)
gmte = 1;                   % Grid margin outside of training points
grid_minte = grid_min; grid_maxte = grid_max;
Ndte = floor(nthroot(Nte,E)); % Nte = Ndte^E;
Xte = ndgridj(grid_minte, grid_maxte,Ndte*ones(E,1)) ;
Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte);


%% Generate Training data
disp('Generate Training Data...')

ntr = floor(Ttr/dttr); Nx0tr = size(x0tr,2); Xtr = zeros(E,ntr,Nx0tr); dXtr = zeros(size(Xtr));
for n = 1:size(x0tr,2)
    [t,xtr] = ode45(dynt,0:dttr:Ttr,x0tr(:,n)');xtr = xtr';
    dXtr(:,:,n) = (xtr(:,2:end)-xtr(:,1:end-1))/dttr + mvnrnd(zeros(E,1),diag(sn.^2),ntr)';
    Xtr(:,:,n) = xtr(:,1:end-1);
end
dXtr = reshape(dXtr,E,ntr*Nx0tr); Xtr = reshape(Xtr,E,ntr*Nx0tr);
dXtr = [dXtr zeros(E,1)]; Xtr = [Xtr zeros(E,1)]; Ytr = Xtr+dXtr;

Ntr = size(Xtr,2);


%% Learn GPSSM
disp('Learn GPSSM...')

[mufun,varfun,gprModels] = learnGPR(Xtr,dXtr,optGPR{:}); 

% Compute gain numerically (approx.)
stdmax = max(sqrt(sum(varfun(Xte).^2,1)));
mumax = max(sqrt(sum(mufun(Xte).^2,1)));
kc = (umax-mumax)/stdmax;


%% Compute Lyapunov Function
disp('Compute Lyapunov Function...')

Vint = Vvar(varfun,grid_min,grid_max,optVvar);
V = @(xi) Vint(xi(1,:)',xi(2,:)');

%% Generate Trajectories
disp('Run Simulation...')

fun = @(t,x) dynt(t,x)-mufun(x)-kc*gradestj(V,x,epsfd);
[Ttraj,Xtraj,te,ye,ie]  = ode45(fun,0:dtsim:Tsim,x0sim,options);

%% Visualization
disp('Visualize...')

figure; hold on; axis tight;
title('GPSSM: Mean and Variance Prediction')
surf(Xte1,Xte2,reshape(sqrt(sum(varfun(Xte).^2,1)),Ndte,Ndte)-1e4,'EdgeColor','none','FaceColor','interp'); colormap(flipud(parula))
Xte_vec = mufun(Xte);
h = streamslice(Xte1,Xte2,reshape(Xte_vec(1,:),Ndte,Ndte),reshape(Xte_vec(2,:),Ndte,Ndte),2,'r'); set( h, 'Color', 'r' )

figure; hold on; axis tight
title('Uncertainty-based Control Lyapunov function and Trajectories')
surf(Xte1,Xte2,-reshape(V(Xte),Ndte,Ndte),'EdgeColor','none','FaceColor','interp');
plot(Xtraj(:,1),Xtraj(:,2),'r');
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'k');%,'AutoScale','off'
%%
disp('Pau');
