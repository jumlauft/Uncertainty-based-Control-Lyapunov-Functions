function  [value,isterminal,direction]  = isconverged(~,x,r)
%ISCONVERGED ode45 event to stop simulation if state is converged
%   Stops simulation if |x|<r
% In:
%       t   1 x 1           not used
%       x   E x 1(1 x E)    state
%       r  1 x 1            radius (proximity to origin
% Out: see ode documentation
% E: Dimensionality of state
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

value =  double(norm(x)<r);
isterminal =1;
direction = 0;
end

