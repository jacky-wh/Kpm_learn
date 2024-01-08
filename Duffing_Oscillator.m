clear all
close all
addpath('./Resources')

%% *************************** Dynamics ***********************************

f_u =  @(t,x,u)(-[ x(2,:); x(1,:)-x(1,:).^3-0.5*x(2,:)+ u] );
n = 2;
m = 1; % number of control inputs

%% ************************** Discretization ******************************
deltaT = 0.1;
%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );

%% ************************** Basis functions *****************************

basisFunction = 'rbf';
% RBF centers
Nrbf = 100;
cent = rand(n,Nrbf)*2 - 1;
rbf_type = 'thinplate'; 
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );
Nlift = Nrbf + n;

%% ************************** Collect data ********************************
tic
disp('Starting data collection')
Nsim = 30;
Ntraj = 1000;

% Random forcing
Ubig = 2*rand([Nsim Ntraj]) - 1;

% Random initial conditions
Xcurrent = (rand(n,Ntraj)*2 - 1);

X = []; Y = []; U = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    X = [X Xcurrent];
    Y = [Y Xnext];
    U = [U Ubig(i,:)];
    Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %1.2f s \n', toc);

%% ******************************* Lift ***********************************

disp('Starting LIFTING')
tic
Xlift = liftFun(X);
Ylift = liftFun(Y);
fprintf('Lifting DONE, time = %1.2f s \n', toc);

%% ********************** Build predictor *********************************

disp('Starting REGRESSION')
tic
W = [Ylift ; X];
V = [Xlift; U];
VVt = V*V';
WVt = W*V';
M = WVt * pinv(VVt); % Matrix [A B; C 0]
Alift = M(1:Nlift,1:Nlift);
Blift = M(1:Nlift,Nlift+1:end);
Clift = M(Nlift+1:end,1:Nlift);

fprintf('Regression done, time = %1.2f s \n', toc);

%% *********************** Predictor comparison ***************************
Tmax = 3;
Nsim = Tmax/deltaT;
u_dt = @(i)((-1).^(round(i/2))); % control signal

% Initial condition
x0 = [0.5;0];
x_true = x0;

% Lifted initial condition
xlift = liftFun(x0);

% Simulate
for i = 0:Nsim-1
    % Koopman predictor
    xlift = [xlift, Alift*xlift(:,end) + Blift*u_dt(i)]; % Lifted dynamics
    
    % True dynamics
    x_true = [x_true, f_ud(0,x_true(:,end),u_dt(i)) ];
    
end
x_koop = Clift * xlift; % Koopman predictions

%% ****************************  Plots  ***********************************
% 
% lw = 4;
% 
% figure
% plot([0:Nsim-1]*deltaT,u_dt(0:Nsim-1),'linewidth',lw); hold on
% title('Control input $u$', 'interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% 
% figure
% plot([0:Nsim]*deltaT,x_true(2,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(2,:), '--r','linewidth',lw)
% axis([0 Tmax min(x_koop(2,:))-0.15 max(x_koop(2,:))+0.15])
% title('Predictor comparison - $x_2$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','location','southwest');
% set(LEG,'interpreter','latex')
% 
% figure
% plot([0:Nsim]*deltaT,x_true(1,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(1,:), '--r','linewidth',lw)
% axis([0 Tmax min(x_koop(1,:))-0.1 max(x_koop(1,:))+0.1])
% title('Predictor comparison - $x_1$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','location','southwest');
% set(LEG,'interpreter','latex')

%% ********************** Feedback control ********************************
% disp('Press any key for feedback control')
% pause

Tmax = 20; % Simlation legth
Nsim = Tmax/deltaT;
REF = 'con'; % 'step' or 'cos'
switch REF
    case 'step'
        ymin = -0.6;
        ymax = 0.6;
        x0 = [0;0.6];
        yrr = 0.3*( -1 + 2*([1:Nsim] > Nsim/2)  ); % reference
    case 'cos'
        ymin = -0.4;
        ymax = 0.4;
        x0 = [-0.1;0.1];
        yrr = 0.5*cos(2*pi*[1:Nsim] / Nsim); % reference
    case 'con'
        ymin = [-3;-2.5];
        ymax = [1;1];
        x0 = [0.5;0];
        yrr =  [-1.25;0]; % reference
        yrr=repmat(yrr,1,Nsim);
end

% Define Koopman controller
C = zeros(2,Nlift); C(1:2,1:2) = [1 0;0 1];
% Weight matrices
Q = diag([1 1]);
R = 0.5;
% Prediction horizon
Tpred = 1;
Np = round(Tpred / deltaT);
% Constraints
xlift_min = [ymin ; nan(Nlift-2,1)];
xlift_max = [ymax ; nan(Nlift-2,1)];

% Build Koopman MPC controller
koopmanMPC  = getMPC(Alift,Blift,C,0,Q,R,Q,Np,-1, 1, xlift_min, xlift_max,'qpoases');

% Closed-loop simultion start
UU_koop1 = [];

XX_koop = x0; UU_koop = [];
x_koop=x0; 
for i = 0:Nsim-1
    if(mod(i,10) == 0)
        fprintf('Closed-loop simulation: iterate %i out of %i \n', i, Nsim)
    end
    % Current value of the reference signal
    yr =[-1.25;0];
    % Koopman MPC
    xlift = liftFun(x_koop); % Lift
    u_koop = koopmanMPC(xlift,yr); % Get control input
    x_koop = f_ud(0,x_koop,u_koop); % Update true state

    % Store values
    XX_koop = [XX_koop x_koop];
    UU_koop = [UU_koop u_koop];
end

%% ****************************  Plots  ***********************************
lw = 4;
for i=1:2
    figure
    yline(yr(i)); hold on
    plot([0:Nsim]*deltaT,XX_koop(i,:), '--r','linewidth',lw)

    % axis([0 Nsim*deltaT -1 1])
    title('Predictor comparison - $x_1$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
    set(gca,'fontsize',20)
    LEG = legend('True','Koopman','Local at $x_0$','Local at 0','location','southwest');
    set(LEG,'interpreter','latex')
end

