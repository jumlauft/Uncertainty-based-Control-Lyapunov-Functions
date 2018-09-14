function [dVdx] = gradestj(fun,x0, eps)
% GRADESTJ Computes numerical gradient of scalar function
% In:
%    x0      E x N      Point where gradient computed
%    fun     fhandle    function  E x 1 -> scalar
%    eps     scalar     distance between two points for slope calculation
% Out:
%  dVdx      E x N
%{
clear, close all,clc; rng default; addpath(genpath('./mtools'));
E = 2; N = 3; x0 = rand(E,N);  fun = @(x) exp(-sum(x.^2,1));
dVdx = gradestj(fun,x0)
dVdxA = -2*x0.*exp(-sum(x0.^2,1))
%}
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

[E, N] = size(x0);
% Check inputs
if ~exist('eps','var'), eps = 1e-3; end
if any(any(isnan(x0))), warning('NaN occured');   end

% Setup evalution points
xpme = repmat(x0,1,E,2) + kron(eye(E),ones(1,N)).*permute([eps -eps],[1 3 2]);

% evaluate function
V = fun(reshape(xpme,E,E*N*2))';

% Compute finite differences
dVdx = reshape((V(1:E*N) - V(N*E+1:end))./(2*eps),N,E)';

% Check output
if any(any(isnan(dVdx))),   warning('NaN occured');   end

end
