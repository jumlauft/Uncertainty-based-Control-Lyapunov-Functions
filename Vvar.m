function V = Vvar(varfun,grid_min,grid_max,opt)
%VVAR Computes the uncertainty-based Lyapunov function
%  For a given variance (varfun) the shortest path to the origin is
%  computed from every point in the grid. Returns the value function
% In:
%     varfun      fhandle(E x N -> E x N)
%       grid_min    E  x 1   lower bounds for grid
%       grid_max    E  x 1   upper bounds for grid
%     opt
%       InterpMethod
%       Ndgrid      1  x 1   Number of grid points per dimension
%       xEq         E  x 1
% Out:
%     V         fhandle     griddedInterpolant
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

% Verify input dimension
E = numel(grid_min);
if numel(grid_max) ~=E, error('wrong input dimensions'); end

% check optional input arguments
if isfield(opt,'Ndgrid'), Ndgrid = opt.Ndgrid; else, Ndgrid = 1e4;end
if isfield(opt,'xEq'), xEq = opt.xEq; else, xEq = zeros(E,1);end
if isfield(opt,'InterpMethod'), InterpMethod = opt.InterpMethod; else, InterpMethod = 'linear';end



Ngrid = Ndgrid^E;
sdfun = @(x) sqrt(varfun(x));
cost = @(x) sqrt(sum(sdfun(x).^2,1));

% Create grid and ensure one gridpoint is at origin/xEq
Xgrid = ndgridj(grid_min,grid_max,Ndgrid*ones(E,1));
[~,i0] = min(sum((Xgrid-xEq).^2,1));
Xgrid = Xgrid - (Xgrid(:,i0)-xEq);
dgrid = sqrt(sum(((grid_max-grid_min)/(Ndgrid-1)).^2,1));

% Setup graph
costgrid = cost(Xgrid);
if Ngrid <= 1e4  % Chose computational more efficient method
    dall = pdist2(Xgrid', Xgrid'); dall(dall>=dgrid) = 0;
    G = sparse((costgrid+costgrid')/2.*dall);
else
    G = sparse(Ngrid,Ngrid);
    for n = 1:Ngrid
        d = sqrt(sum((Xgrid(:,n)-Xgrid).^2,1));
        ii = find(d<dgrid);
        avgcost = (costgrid(ii)+costgrid(n))/2;
        G = G + sparse(ii,n*ones(1,numel(ii)),d(ii).*avgcost,Ngrid,Ngrid);
    end
end

% Compute distance to origin in graph
[dist] = graphshortestpath(G,i0);
dist = reshape(dist,Ndgrid,Ndgrid)';

% Stetup Interpolation
Xgrid1 = reshape(Xgrid(1,:),Ndgrid,Ndgrid)'; Xgrid2 = reshape(Xgrid(2,:),Ndgrid,Ndgrid)';
V = griddedInterpolant(Xgrid1,Xgrid2,dist,InterpMethod);


end
