function [Xscaled,meanX,stdX] = autoscale_new(X,meanX,stdX)
% -------------------------------------------------
% ***  Autoscale  *** 
% Coded by Manabu KANO, Kyoto Univ., Feb. 27, 2000
%                     last updated : Aug.  8, 2001
%
% USAGE : 
% [Xscaled,meanX,stdX] = autoscale(X,meanX,stdX)
%
% DESCRITION :
% This function autoscales X.  The mean and the standard
% deviation of Xscaled are 0 and 1, or X is scaled by 
% using meanX and stdX.
%
% --- Input ---
% X		: data matrix (samples*variables)
%   ( option )
% meanX		: means of variables
% stdX		: standard deviations of variables
%		  If meanX or stdX is empty '[]', 
% 		  then X is not scaled.
% 
% --- Output --- 
% Xscaled	: autoscaled X
% meanX		: means of variables
% stdX		: standard deviations of variables
% 
% ----------------------------------------------------


if nargin < 3, stdX = std(X); end
if isempty(stdX)==1,  stdX  = ones(1,size(X,2));  end
if nargin < 2, meanX = mean(X); end
if isempty(meanX)==1, meanX = zeros(1,size(X,2)); end

rowX = size(X,1);

std0 = find(stdX<=10^(-10));
for i=1:length(std0)
	stdX(1,std0(i)) = inf;
end
Xscaled = (X-meanX(ones(rowX,1),:))./stdX(ones(rowX,1),:);

for i=1:length(std0)
    Xscaled(1,std0)=0;
end

for i=1:length(std0)
	stdX(1,std0(i)) = 0;
end
%
% Copyright (c) 2000-. Manabu KANO
% All rights reserved.
% e-mail: kano@cheme.kyoto-u.ac.jp
% http://www-pse.cheme.kyoto-u.ac.jp/~kano
%
