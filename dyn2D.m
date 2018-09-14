function [dxdt] = dyn2D(~,x)
%DYN2D time invariant 2D nonlinear time continous dynamics
% In:
%    t      ~     time (not used)
%    x    2 x N     state
% Out:
%    dxdt 2 x N     state derivative
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

dxdt(1,:) = +x(1,:) +(-1+cos(x(1,:))).*x(2,:);
dxdt(2,:) = sigmf(x(1,:),[2 0])-0.5 +x(2,:);

