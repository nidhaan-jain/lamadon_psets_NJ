% This file backs out the Hessian of the utility function from Frisch 
% elasticities for the average consumption and hours. 

clear
clc

%% Parameters
% Elasticities
eta_cp=-0.417;       % note that need to plug-in the negative of the elasticity from the estimation
eta_cw1=-0.162;
eta_cw2=-0.050;
eta_h1p=0.126;
eta_h1w1=0.681;
eta_h1w2=0.159;
eta_h2p=0.079;
eta_h2w1=0.325;
eta_h2w2=0.958;

% Quantities (from desc. stats)
c_bar = 39257;
y_bar = 67008; 
yw_bar= 32988;

h1_bar=2302;    
h2_bar=1688;    

% Prices 
p=1;
w1=y_bar/h1_bar;  
w2=yw_bar/h2_bar;   

% The Frisch demand matrix
Fd=[eta_cp*c_bar/p,eta_cw1*c_bar/w1,eta_cw2*c_bar/w2;...
    -eta_h1p*h1_bar/p,-eta_h1w1*h1_bar/w1,-eta_h1w2*h1_bar/w2;...
    -eta_h2p*h2_bar/p,-eta_h2w1*h2_bar/w1,-eta_h2w2*h2_bar/w2];

format short
eig(Fd)
Fd
Hessian=inv(Fd)
