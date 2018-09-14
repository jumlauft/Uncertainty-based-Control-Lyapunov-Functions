function [GPm,GPs2, gprMdls, sn2] = learnGPR(X,Y, varargin)
%%LEARNGPR Buildes mean and variance function for GP incl. hyp opt.
% In:
%     X        D  x Ntr   Input training points
%     Y        E  x Ntr   Output training points
%    varargin             options, see doc fitrgp
% Out:
%     GPm     fhandle      E x N -> D x N   mean function
%     GPs2    fhandle      E x N -> D x N   variance function
%     gprMdls struct
%     sn2     E x 1        observation noise sn2
% E: Dimensionality of outputs
% D: Dimensionality of inputs
% Ntr: Number of training points
% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2018-09

E = size(Y,1);
optGPR = cell(E,1);   [optGPR{:}] = deal(varargin);

% Expand input arguments to multiple dimensions
for e = 1:E
    for l=1:length(varargin)
        if strcmp(varargin{l},'KernelParameters'), optGPR{e}{l+1} = varargin{l+1}(:,e);end
        if strcmp(varargin{l},'Sigma'), optGPR{e}{l+1} = varargin{l+1}(e); end
        if strcmp(varargin{l},'Beta'), optGPR{e}{l+1} = [0; varargin{l+1}(:,e)]; end 
    end
end

% Initialize and train GP models
gprMdls = cell(E,1); sn2 = zeros(E,1);
for e = 1:E
    gprMdls{e} = fitrgp(X',Y(e,:)',optGPR{e}{:});
    sn2(e) = gprMdls{e}.Sigma.^2;
end

% Setup output functions
GPm = @(x) GPmfun(x,gprMdls);
GPs2 = @(x) GPs2fun(x,gprMdls);
end

% Mean Prediction
function m = GPmfun(x,gprMdls)
E = size(gprMdls,1); N = size(x,2); m = zeros(E,N);
for e=1:E
    m(e,:) = predict(gprMdls{e},x');
end
end

% Variance prediciton
function s2 = GPs2fun(x,gprMdls)
E = size(gprMdls,1); N = size(x,2); sd = zeros(E,N);
for e=1:E
    [~, sd(e,:)] = predict(gprMdls{e},x');
end
s2 = sd.^2;
end
