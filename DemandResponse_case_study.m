%% Advanced Methods for the Optimal Management of the Electrical Grid
%% *Case Study 1*
%% *Initialize Program*

clear all% all variables in the workspace are deleted
clc % the commands printed in the command windows are deleted
close all %close all plot windows
% Define Progam Parameters

compute_centralized = [0 0 0 0]'; %array containing the activation of the centralized solution computing for each run
N_arr=[3000 4000 5000 10000]'; %array containing the number of prosumers for each case

%compute_centralized = [0];
%N_arr = [32];
M=96; %daily time-slots
taus=24/M; %time-slot duration
tr1=6.75/taus; %time-slot at which the first request arrives
t01=tr1+1; %time-slot at which the request start to be satisfied
tf1=9/taus; %time-slot at which the req stops
% Create vector representing the time instants in which the TSO request is
% active
dtso = zeros(1,M);
for t=t01:tf1
    dtso(1,t) = 1;
end
%gamma1 = 18*N;
gamma1 = 300;
epsilon = 0.05;

% DISTRIBUTED ALGO PARAMETERS
%alpha1_n = 0.0035;
alpha1_n = 0.0035;
%alpha1_man = 1e-5;
alpha2_exp = 0.3;
alpha2 = 0.51;
%alpha2 = 1;
w = 10;
% Initialize variables to store results of each run

x_cen = {zeros(length(N_arr))};
fval_cen = {zeros(length(N_arr))};
time_cen = {zeros(length(N_arr))};

x_fal = {zeros(length(N_arr))};
fval_fal = {zeros(length(N_arr))};
epsilon_global = {zeros(length(N_arr))};
ko_global = {zeros(length(N_arr))};
ki_global = {zeros(length(N_arr))};
time_fal = {zeros(length(N_arr))};
time_fal_in = {zeros(length(N_arr))};
Kext_fal = {zeros(length(N_arr))};

x_vuj = {zeros(length(N_arr))};
fval_vuj = {zeros(length(N_arr))};
ko_global_v = {zeros(length(N_arr))};
ki_global_v = {zeros(length(N_arr))};
costFal_global = {zeros(length(N_arr))};
costVuj_global = {zeros(length(N_arr))};
time_vuj = {zeros(length(N_arr))};
time_vuj_in = {zeros(length(N_arr))};
Kext_vuj = {zeros(length(N_arr))};


% Define constraints, costs and reference profiles for each run randomly (but respecting constraints)

maxPpl_global = {zeros(length(N_arr))};
dpl_ran_global = {zeros(length(N_arr))};
Ppl_ref_global = {zeros(length(N_arr))};
Epl_global = {zeros(length(N_arr))};

maxS_global = {zeros(length(N_arr))};
minS_global = {zeros(length(N_arr))};
S0_global = {zeros(length(N_arr))};
maxPb_global = {zeros(length(N_arr))};
Pb_ref_global = {zeros(length(N_arr))};


maxPg_global = {zeros(length(N_arr))};
minPg_global = {zeros(length(N_arr))};
Pg_ref_global = {zeros(length(N_arr))};


P_ref_global = {zeros(length(N_arr))};

Cg_global = {zeros(length(N_arr))};
Cb_global = {zeros(length(N_arr))};
Cpl_global = {zeros(length(N_arr))};

for n=1:length(N_arr)
    N = N_arr(n);
    disp("Profile with " + N + " prosumers");

    % FOR THE PROGRAMMABLE LOAD
    rng('shuffle');
    maxPpl_array = rand(N,1)*90+10; %random generated programmable max loads power
    maxPpl = zeros(N,M); %kept constant on the M time slots
    for t=1:M
        maxPpl(:,t) = maxPpl_array;
    end
    maxPpl_global{n} = maxPpl;
    rng('shuffle');
    dpl_ran = randi(4, N, M/8); %random integer matrix from 1 to 4
    dpl_ran_global{n} = dpl_ran;
    dpl_ref = zeros(N,M);
    for ti=1:M/8
        for t=ti*8-8+1:ti*8
            dpl_ref(:, t) = dpl_ran(:, ti);
        end
    end
    Ppl_ref = (dpl_ref/4).*maxPpl; %random generated pre-agreed load profile
    Ppl_ref_global{n} = Ppl_ref;
    Epl=sum(taus*Ppl_ref,2);
    Epl_global{n} = Epl;


    % FOR THE BATTERY STORAGE SYSTEM
    maxS = Epl/4; %max state of charge
    maxS_global{n} = maxS;
    minS = 0.1*maxS; %min state of charge
    minS_global{n} = minS;
    rng('shuffle');
    S0 = rand(N,1).*(maxS-minS) + minS; %initial state of charge of the battery
    S0_global{n} = S0;
    maxPb_array = maxS/(taus*M/4); %max power of the battery (max charging)
    maxPb = zeros(N,M); %kept constant on the M time slots
    for i=1:M
        maxPb(:,i) = maxPb_array;
    end
    maxPb_global{n} = maxPb;


    % FOR THE GENERATOR
    rng('shuffle');
    dg_ran = randi(5,N,M/16);
    dg_ref = zeros(N,M);
    for ti=1:M/16
        for t=ti*16-16+1:ti*16
            dg_ref(:, t) = dg_ran(:, ti);
        end
    end
    Gcoeff = (dg_ref/4 + 0.25);
    Pg_ref = Ppl_ref.*Gcoeff;
    maxPg = maxPpl*1; %max power of the generators
    maxPg_global{n} = maxPg;
    minPg = 0.2*maxPg; %min power of the generators
    minPg_global{n} = minPg;
    e=0;
    % Check if the profile violates the constraints and correct it
    for t=1:M
        for n_i = 1:N
            if Pg_ref(n_i,t) > maxPg(n_i,t)
                Pg_ref(n_i,t) = maxPg(n_i,t);
                %disp("Found Violation in generation profile (+)");
                e=e+1;
            else 
                if Pg_ref(n_i,t) < minPg(n_i,t)
                    Pg_ref(n_i,t) = minPg(n_i,t);
                    %disp("Found Violation in generation profile (-)");
                    e=e+1;
                end
            end
        end
    end
    disp("    - Number of violations in the generator profile: "+e);

    Pg_ref_global{n} = Pg_ref;


    % FOR THE WHOLE SYSTEM
    rng('shuffle');
    P_ref = Pg_ref - Ppl_ref; %start by balancing generation and load
    % add some random quantity (compatible with battery power constraints) to the refernce profile
    for i=1:N    
        added_ref=zeros(1,M);
        for ti=1:M/4
            additive = rand(1,M/4);
            signvar = randi(2,1,M/4);
            for t=ti*4-4+1:ti*4
                added_ref(:,t) = additive(:,ti).*maxPb(i,t).*(-1^(signvar(:,ti)));
            end
        end
        P_ref(i,:) = P_ref(i,:) + added_ref/10;
    end
    % Compute the battery reference profile
    Pb_ref = P_ref - Pg_ref + Ppl_ref;
    % Check if the SOC constraints are violated and correct
    battery_cap_viol_sign = 0;
    battery_cap_viol_mod = 0;
    for i =1:N
        for t =1:M
            if (minS(i) > S0(i) - taus*sum(Pb_ref(i,1:t),2))||(S0(i) - taus*sum(Pb_ref(i,1:t),2) > maxS(i))
                battery_cap_viol_sign = battery_cap_viol_sign + 1 - battery_cap_viol_mod;
                %disp("Battery capacity violated, changed sign")
                Pb_ref(i,t) = -Pb_ref(i,t)/10;
            end
            if (minS(i) > S0(i) - taus*sum(Pb_ref(i,1:t),2))||(S0(i) - taus*sum(Pb_ref(i,1:t),2) > maxS(i))
                battery_cap_viol_mod = battery_cap_viol_mod + 1;
                %disp("Battery capacity violated, changed module")
                Pb_ref(i,t) = Pb_ref(i,t)/10;
            end
            if (minS(i) > S0(i) - taus*sum(Pb_ref(i,1:t),2))||(S0(i) - taus*sum(Pb_ref(i,1:t),2) > maxS(i))
                %disp("Battery capacity violated, changed module")
                Pb_ref(i,t) = 0;
            end
        end
    end
    Pb_ref_global{n} = Pb_ref;
    disp("Battery reference profile constraints violations:")
    disp("  - Sign: "+battery_cap_viol_sign)
    disp("  - Module: "+battery_cap_viol_mod)
    
    P_ref = Pg_ref - Ppl_ref + Pb_ref;
    P_ref_global{n} = P_ref;

    % COSTS
    rng('shuffle');
    Cg_array = rand(N,1); %unitary cost of the energy produced by the generator 
    Cb_array = rand(N,1); %unitary cost of the aging of the battery
    Cpl_array = rand(N,1)*35; %Unitary cost for changes in the programmable load consumption profile
    Cg = zeros(N,M); %kept constant on the M time slots
    Cb = zeros(N,M);
    Cpl = zeros(N,M);
    for t=1:M
        Cg(:,t) = Cg_array;
        Cb(:,t) = Cb_array;
        Cpl(:,t) = Cpl_array;
    end
    Cg_global{n} = Cg;
    Cb_global{n} = Cb;
    Cpl_global{n} = Cpl;


    disp("------------------------------------------------------------------------------");

end


%% Run Algorithms and gather results

for n=1:length(N_arr)
    N = N_arr(n);
    gamma1 = 18*N;
    alpha1 = alpha1_n/N;
    if N<11
        alpha1 = 1e-20;
        alpha2 = 0.81;
    end
    if N<21
        alpha1 = 1e-15;
        alpha2 = 0.80;
    end
    if N<31
        alpha1 = 1e-10;
        alpha2 = 0.70;
    end
    if N<71
        alpha1 = 1e-5;
        alpha2 = 0.65;
    end
    % if N<15
    %     alpha1 = 1e-20;
    %     alpha2 = 0.95;
    % end

    % run centralized algo if wanted
    epsilon = 0.05;
    if compute_centralized(n) == 1
        tic
        [x_opt, fval] = centralized_algo(N, M, maxPpl_global{n}, Epl_global{n}, maxPb_global{n}, S0_global{n}, minS_global{n}, maxS_global{n}, minPg_global{n}, maxPg_global{n}, Pg_ref_global{n}, Pb_ref_global{n}, Ppl_ref_global{n}, P_ref_global{n}, gamma1, epsilon, Cg_global{n}, Cb_global{n}, Cpl_global{n}, dtso, taus, tr1, tf1);
        tf_exe = toc;
        time_cen{n} = tf_exe;
        x_cen{n} = x_opt;
        fval_cen{n} = fval;
    end
    % run falsone et al distributed algo
    epsilon = 0.05;
    out = 1;
    Kext = 0;
    while (out == 1)
            tic
            [x_opt, fval, ki_tot, ko, epsilon_f, t_exe_in, out] = decentralized_fal(N, M, maxPpl_global{n}, Epl_global{n}, maxPb_global{n}, S0_global{n}, minS_global{n}, maxS_global{n}, minPg_global{n}, maxPg_global{n}, Pg_ref_global{n}, Pb_ref_global{n}, Ppl_ref_global{n}, P_ref_global{n}, dtso, gamma1, epsilon, Cg_global{n}, Cb_global{n}, Cpl_global{n}, alpha1_n, alpha2, w, taus, tr1, tf1);
            tf_exe = toc;
            if out == 1
                alpha2 = alpha2 * 0.75;
                %alpha1_n = alpha1_n * 0.75;
                if alpha2<0.51
                    alpha2 = 0.51;
                end
            end
        Kext = Kext + 1;
        epsilon = epsilon * 1.5;
    end
    x_fal{n} = x_opt;
    fval_fal{n} = fval;
    epsilon_global{n} = epsilon_f;
    ko_global{n} = ko;
    ki_global{n} = ki_tot;
    time_fal{n} = tf_exe;
    time_fal_in{n} = t_exe_in;
    Kext_fal{n} = t_exe_in;

    % run vujanic et al distributed algo
    epsilon = 0.05;
    out = 1;
    while (out == 1)
            tic
            [x_opt, fval, ki_tot, ko, epsilon_v, t_exe_in, out] = decentralized_vuj(N, M, maxPpl_global{n}, Epl_global{n}, maxPb_global{n}, S0_global{n}, minS_global{n}, maxS_global{n}, minPg_global{n}, maxPg_global{n}, Pg_ref_global{n}, Pb_ref_global{n}, Ppl_ref_global{n}, P_ref_global{n}, dtso, gamma1, epsilon, Cg_global{n}, Cb_global{n}, Cpl_global{n}, alpha1_n, alpha2, taus, tr1, tf1);
            tf_exe = toc;
            if out == 1
                alpha2 = alpha2 * 0.75;
                %alpha1_n = alpha1_n * 0.75;
                if alpha2<0.51
                    alpha2 = 0.51;
                end
            end
        Kext = Kext + 1;
        epsilon = epsilon * 1.5;
    end
    x_vuj = x_opt;
    fval_vuj = fval;
    ko_global_v = ko;
    ki_global_v = ki_tot;
    time_vuj{n} = tf_exe;
    time_vuj_in{n} = t_exe_in;
    filename = "DRProblem_"+N+"_prosumers";
    save(filename);

end
beep
pause(1)
beep
pause(1)
beep
%% Centralized algorithm

function [x_opt, fval] = centralized_algo(N, M, maxPpl, Epl, maxPb, S0, minS, maxS, minPg, maxPg, Pg_ref, Pb_ref, Ppl_ref, P_ref, gamma1, epsilon, Cg, Cb, Cpl, dtso, taus, tr1, tf1)
    DRprob = optimproblem;

    % OPTIMIZATION VARIABLES
    
    Ppl = optimvar('Ppl',N, M,'LowerBound',0); % Programmable Load Power
    dpl = optimvar('dpl',N, M,'Type','integer','LowerBound',0,'UpperBound',4); % programmable load level
    
    Pb = optimvar('Pb',N, M); % battery power
    
    Pg = optimvar('Pg',N, M,'LowerBound',0); %generator power
    dg = optimvar('dg',N, M,'Type','integer','LowerBound',0,'UpperBound',1); %is generator gi on or off at time-slot j?


    hb = optimvar('hb',N, M);   %auxiliary variable to make the cost function linear
    hpl = optimvar('hpl',N, M); %auxiliary variable to make the cost function linear

    
    % CONSTRAINTS
    % Programmable Load
    DRprob.Constraints.conspl1 = Ppl == (dpl/4).*maxPpl;
    DRprob.Constraints.conspl2 = Epl == sum(taus*Ppl,2);

    % Battery
    minPb = -maxPb; %min power of the battery (max discharging)
    consb1 = minPb <= Pb;
    consb2 = Pb <= maxPb;
    consb3 = minPb <= Pb; %random initialization of constraint vector
    consb4 = Pb <= maxPb; %random initialization of constraint vector
    for t=1:M
        consb3(:,t) = minS <= S0 - taus*sum(Pb(:,1:t),2);
        consb4(:,t) = S0 - taus*sum(Pb(:,1:t),2) <= maxS;    
    end
    DRprob.Constraints.consb1 = consb1;
    DRprob.Constraints.consb2 = consb2;
    DRprob.Constraints.consb1 = consb3;
    DRprob.Constraints.consb2 = consb4;

    % Generator
    consg1 = dg.*minPg <= Pg;
    consg2 = Pg <= dg.*maxPg;
    DRprob.Constraints.consg1 = consg1;
    DRprob.Constraints.consg2 = consg2;

    % Whole System
    DRprob.Constraints.cons1 = Pg(:, 1:tr1) == Pg_ref(:, 1:tr1);
    DRprob.Constraints.cons2 = Pb(:, 1:tr1) == Pb_ref(:, 1:tr1);   %you need to keep everything as it was since
    DRprob.Constraints.cons3 = Ppl(:, 1:tr1) == Ppl_ref(:, 1:tr1); %before tr1 you don't even know the request
    % DRprob.Constraints.cons4 = cat(2,Pg(:, tf1+1:tr2),Pg(:, tf2+1:M)) + cat(2,Pb(:, tf1+1:tr2),Pb(:, tf2+1:M)) - cat(2,Ppl(:, tf1+1:tr2),Ppl(:, tf2+1:M)) == cat(2,P_ref(:, tf1+1:tr2),P_ref(:, tf2+1:M));
    DRprob.Constraints.cons4 = Pg(:, tf1+1:M) + Pb(:, tf1+1:M) - Ppl(:, tf1+1:M) == P_ref(:, tf1+1:M);
    
    
    consCoup1 = (1-epsilon*sign(gamma1))*gamma1*dtso <= sum((Pg + Pb - Ppl),1) - sum(P_ref,1);
    consCoup2 = sum((Pg + Pb - Ppl),1) - sum(P_ref,1) <= (1+epsilon*sign(gamma1))*gamma1*dtso;
    DRprob.Constraints.consCoup1 = consCoup1; %coupling constraint
    DRprob.Constraints.consCoup2 = consCoup2; %coupling constraint

    % Auxiliary Constraints
    consAux1 = minPb <= Pb;
    consAux2 = minPb <= Pb;
    consAux1(:,1) = zeros(N,1) <= hb(:,1);
    consAux2(:,1) = zeros(N,1) <= hb(:,1);
    for t=2:M  %at time 1 we don't have memory of the past
        consAux1(:,t) = Pb(:,t) - Pb(:,t-1) <= hb(:,t);
        consAux2(:,t) = -Pb(:,t) + Pb(:,t-1) <= hb(:,t);
    end
    consAux3 = Ppl - Ppl_ref <= hpl;
    consAux4 = -Ppl + Ppl_ref <= hpl;
    DRprob.Constraints.consAux1 = consAux1;
    DRprob.Constraints.consAux2 = consAux2;
    DRprob.Constraints.consAux3 = consAux3;
    DRprob.Constraints.consAux4 = consAux4;

    % COST FUNCTION
    cost = sum(sum(Cg.*Pg + Cb.*hb + Cpl.*hpl,2),1);
    DRprob.Objective = cost;

    % SOLVE PROBLEM
    options = optimoptions(@intlinprog, "HeuristicsMaxNodes", 100, "Heuristics",'advanced');
    [x_opt,fval] = solve(DRprob,"Options",options);


    x = x_opt;

    algo_str = "Centralized";
    previousgraph = [];
    figure
    sgtitle(["Power profiles " + N + " prosumers", algo_str],'fontweight','bold')
    for i=1:3
        rng('shuffle');
        pros = randi(N);
        while ismember(pros, previousgraph) ~= 0
            pros = randi(N);
        end
        previousgraph = [previousgraph pros];
        Pg = x.Pg(pros,:);
        Pb = x.Pb(pros,:);
        Ppl = x.Ppl(pros,:);
        subplot(3,3,i)
        hold on
        plot(Pg);
        plot(Pg_ref(pros,:)'); 
        if i == 1
            ylabel({"Generator Power","",'P_g'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_g'+ "_" + pros)
        end
        xlabel("time")
        title("Prosumer " + pros)
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+3)
        hold on
        plot(Pb);
        plot(Pb_ref(pros,:)');
        if i == 1
            ylabel({"Battery Power","",'P_b'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_b'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+6)
        hold on
        plot(Ppl);
        plot(Ppl_ref(pros,:)');
        if i == 1
            ylabel({"Programmable Load Power","",'P_{pl}'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_p_l'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    end
    drawnow
    savefig(algo_str+"_"+N+"_prosumer_ProsumerProfile"+".fig")

    figure
    title(algo_str + ". " + "Total Profile with " + N + " prosumers")
    hold on
    plot(sum(x.Pg+x.Pb-x.Ppl,1));
    title("Total Profile with " + N + " prosumers")
    subtitle(algo_str)
    plot(sum(P_ref,1));
    lightGreen = [0.85, 1, 0.85];
    patch([1:M fliplr(1:M)], [(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1) fliplr(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1)], lightGreen); 
    plot(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1);
    plot(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1);
    ylabel('P')
    xlabel("time")
    legend("Solution","Reference profile","TSO Request Admissible Region","Profile with TSO request - \epsilon","Profile with TSO request + \epsilon",'Location','southeast')
    hold off
    drawnow
    savefig(algo_str+"_"+N+"_pool_ProsumerProfile"+".fig")

end
%% Distributed algorithm (Falsone et al)

function [x_opt, fval, ki_tot, ko, epsilon_f, t_exe_in, out] = decentralized_fal(N, M, maxPpl, Epl, maxPb, S0, minS, maxS, minPg, maxPg, Pg_ref, Pb_ref, Ppl_ref, P_ref, dtso, gamma1, epsilon, Cg, Cb, Cpl, alpha1_n, alpha2, w, taus, tr1, tf1)

    % DEFINE NEW COST MATRIX
    c_T = [Cg zeros(N,M) zeros(N,M) zeros(N,M) zeros(N,M) Cb Cpl];
    algo_str = "Falsone et al";
    figtitle = algo_str + " with " + N + " prosumers" + "(\epsilon = " + epsilon + ")";
    figure('Name',figtitle);

    % DEFINE Ai MATRIX AND b VECTOR
    Ai1 = zeros(M,7*M);
    Ai2 = zeros(M,7*M);
    Mdtso1 = [];
    Mdtso2 = [];
    Mdtso_1 = [];
    Mdtso_2 = [];
    t_exe_in = [];
    
    for i=1:M
        Mdtso1 = [Mdtso1; dtso];
        Mdtso2 = [Mdtso2  dtso'];
    end
    for i = 1:7
        Mdtso_1 = [Mdtso_1 Mdtso1];
        Mdtso_2 = [Mdtso_2 Mdtso2];
    end
    
    for i=1:M
        for j=1:M
            if (i == j)
                Ai1(i,j) = -1;
                Ai1(i,M+j) = 0;
                Ai1(i,2*M+j) = -1;
                Ai1(i,3*M+j) = 1;
                Ai1(i,4*M+j) = 0;
                Ai1(i,5*M+j) = 0;
                Ai1(i,6*M+j) = 0;
                Ai2(i,j) = 1;
                Ai2(i,M+j) = 0;
                Ai2(i,2*M+j) = 1;
                Ai2(i,3*M+j) = -1;
                Ai2(i,4*M+j) = 0;
                Ai2(i,5*M+j) = 0;
                Ai2(i,6*M+j) = 0;
            end
        end
    end
    
    Ai=[Ai1.*Mdtso_1.*Mdtso_2;Ai2.*Mdtso_1.*Mdtso_2];
    b = [((-(1-epsilon*sign(gamma1))*gamma1-sum(P_ref,1)).*dtso)'
         ((+(1+epsilon*sign(gamma1))*gamma1+sum(P_ref,1)).*dtso)'];

    % ALGORITHM
    ki_arr = [];
    ko_arr = [];
    Kext_arr = [];
    lambda_arr = [];
    dlambda_arr = [];
    rho_arr = [];
    max_v_arr = [];
    epsilon_arr = [];
    relcon_arr = [];
    abscon_arr = [];
    
    
    ki=1;
    ko=1;
    ki2 = 0;
    out = 0;
    v(:, 1) = 10000*ones(2*M, 1);
    lambda(:, 1) = zeros(2*M, 1);
    rho(:, 1) = zeros(2*M, 1);
    deltalambda = 1;

    alpha1 = alpha1_n/N;
    
    while max(v(:, ki)>0) && out==0 %&& ko<15
        ki = 1;
        convergence = 0;
        while (convergence == 0) && (max(v(:, ki)>0))  %|| max(deltalambda ./ lambda_1) >= 1e-2  || a >=0.0002
    
            %PLOT FROM PREVIOUS RUN
            if (ki>1)||(ko>1) %skip first step

                if ki > 1
                    deltalambda_c = lambda(:, ki) - lambda(:, ki-1);
                    lambda_c = lambda(:, ki-1);
                    relcon = max(deltalambda_c./lambda_c);
                    abscon = max(deltalambda_c);
                end
                relcon_arr = [relcon_arr relcon];
                abscon_arr = [abscon_arr abscon];
    
                ki_arr = [ki_arr ki];
                ko_arr = [ko_arr ko];
                lambda_arr = [lambda_arr norm(lambda)];
                rho_arr = [rho_arr norm(rho)];
                if ki > 1
                    max_v_arr = [max_v_arr max(v(:, ki))];
                else
                    max_v_arr = [max_v_arr maxv_show_next];
                end
                epsilon_arr = [epsilon_arr epsilon];
                dlambda_arr = [dlambda_arr max(abs(deltalambda))];
                %Plot current advance, lambda and rho values
                subplot(3, 3, 1)
                yyaxis left
                plot(ki_arr,"-o",'LineWidth',1.5)
                yyaxis right
                hold on
                plot(ko_arr,"-^",'LineWidth',1.5)
                hold off
                title("k_i and k_o")
                legend("k_i","k_o",'Location','southeast')
                
                subplot(3, 3, 2)
                title("|\lambda| and |\rho|")
                yyaxis left
                lmax = max(lambda_arr);
                lmin = min(lambda_arr);
                if (ki>2)||(ko>1)
                    ylim([lmin*0.9 lmax*1.1])
                end
                plot(lambda_arr,"-o",'LineWidth',1.5)
                yyaxis right
                rmax = max(0.0001, max(rho_arr));
                rmin = min(rho_arr);
                ylim([rmin*0.9 rmax*1.1])
                xlim([0 length(rho_arr)])
                plot(rho_arr,"-^",'LineWidth',1.5)
                legend("|\lambda|","|\rho|",'Location','southeast')
    
                subplot(3, 3, 3)
                hold on
                plot(dlambda_arr,"-^",'LineWidth',1.5)
                dlmax = max(dlambda_arr);
                dlmin = min(dlambda_arr);
                ylim([0.9*dlmin dlmax*1.1])
                xlim([0 length(dlambda_arr)])
                set(gca, 'YScale', 'log')
                title("\Delta\lambda Max")
                hold off
    
                if ki > 1
                    subplot(3, 3, [4, 5, 7, 8])
                    hold on
                    plot(max_v_arr,"-^b",'LineWidth',1.5)
                    plot(zeros(length(max_v_arr)),"--g",'LineWidth',1.5)
                    hold off
                    vmax = max(max_v_arr);
                    ylim([-vmax*0.15 vmax*1.1])
                    xlim([0 length(max_v_arr)])
                    title("Max Violation")
                end
                
                subplot(3, 3, 9)
                hold on
                plot(abscon_arr,"-^",'LineWidth',1.5)
                plot(ones(length(abscon_arr))*1e-4,"--g",'LineWidth',1.5)
                set(gca, 'YScale', 'log')
                title("Absolute Convergence")
                acmax = max(abscon_arr);
                acmin = min(abscon_arr);
                if (ki>2)||(ko>1)
                    ylim([0.1*acmin acmax*1.1])
                end
                xlim([0 length(epsilon_arr)])
                hold off
                
                subplot(3, 3, 6)
                hold on
                plot(relcon_arr,"-^",'LineWidth',1.5)
                plot(ones(length(abscon_arr))*1e-3,"--g",'LineWidth',1.5)
                set(gca, 'YScale', 'log')
                title("Relative Convergence")
                rcmax = max(relcon_arr);
                rcmin = min(relcon_arr);
                if (ki>3)||(ko>1)
                    ylim([0.1*rcmin rcmax*1.1])
                end
                xlim([0 length(epsilon_arr)])
                hold off
                drawnow
            end
    
            parfor j=1:N   %find local optimum
                tic
                probj = optimproblem;
                % Programmable Load (variables definition and constraints)
                Pplj = optimvar('Pplj',1, M,'LowerBound',0); %programmable load power
                dplj = optimvar('dplj',1, M,'Type','integer','LowerBound',0,'UpperBound',4);
                probj.Constraints.conspl1 = Pplj == (dplj/4).*maxPpl(j,:);
                probj.Constraints.conspl2 = Epl(j,:) == sum(taus*Pplj,2);
                % Battery (variables definition and constraints)
                Pbj = optimvar('Pbj',1, M); %battery power
                minPb = -maxPb;
                consbj1 = minPb(j,:) <= Pbj;
                consbj2 = Pbj <= maxPb(j,:);
                consbj3 = minPb(j,:) <= Pbj; %random initialization
                consbj4 = Pbj <= maxPb(j,:); %random initialization
                for t=1:M
                    consbj3(:,t) = minS(j,:) <= S0(j,:) - taus*sum(Pbj(:,1:t),2);
                    consbj4(:,t) = S0(j,:) - taus*sum(Pbj(:,1:t),2) <= maxS(j,:);    
                end
                probj.Constraints.consb1 = consbj1;
                probj.Constraints.consb2 = consbj2;
                probj.Constraints.consb1 = consbj3;
                probj.Constraints.consb2 = consbj4;
                % Generator (variables definition and constraints)
                Pgj = optimvar('Pgj',1, M,'LowerBound',0); %generator power
                dgj = optimvar('dgj',1, M,'Type','integer','LowerBound',0,'UpperBound',1); %on-off state
                consgj1 = dgj.*minPg(j,:) <= Pgj;
                consgj2 = Pgj <= dgj.*maxPg(j,:);
                probj.Constraints.consg1 = consgj1;
                probj.Constraints.consg2 = consgj2;
                % Reference profile constraints
                probj.Constraints.cons1 = Pgj(:, 1:tr1) == Pg_ref(j, 1:tr1);
                probj.Constraints.cons2 = Pbj(:, 1:tr1) == Pb_ref(j, 1:tr1);   %you need to keep everything as it was since
                probj.Constraints.cons3 = Pplj(:, 1:tr1) == Ppl_ref(j, 1:tr1); %before tr1 you don't even know the request
                probj.Constraints.cons4 = Pgj(:, tf1+1:M) + Pbj(:, tf1+1:M) - Pplj(:, tf1+1:M) == P_ref(j, tf1+1:M);
                % Auxiliary constraints
                hbj = optimvar('hbj',1, M);   %auxiliary variable to make the cost function linear
                hplj = optimvar('hplj',1, M); %auxiliary variable to make the cost function linear
    
                consAuxj1 = minPb(j,:) <= Pbj;
                consAuxj2 = minPb(j,:) <= Pbj;
                consAuxj1(:,1) = zeros(1,1) <= hbj(:,1);
                consAuxj2(:,1) = zeros(1,1) <= hbj(:,1);
                for t=2:M  %at time 1 we don't have memory of the past
                    consAuxj1(:,t) = Pbj(:,t) - Pbj(:,t-1) <= hbj(:,t);
                    consAuxj2(:,t) = -Pbj(:,t) + Pbj(:,t-1) <= hbj(:,t);
                end
                consAuxj3 = Pplj - Ppl_ref(j,:) <= hplj;
                consAuxj4 = -Pplj + Ppl_ref(j,:) <= hplj;
                probj.Constraints.consAux1 = consAuxj1;
                probj.Constraints.consAux2 = consAuxj2;
                probj.Constraints.consAux3 = consAuxj3;
                probj.Constraints.consAux4 = consAuxj4;
    
                % Objective Function (costs, auxiliary variables and additional constraints)
                costj = c_T(j,:) + lambda(:, ki)'*Ai;
                xj = [Pgj dgj Pbj Pplj dplj hbj hplj]';
                probj.Objective = costj*xj;
    
    
                sj = solve(probj);
                t_exe = toc;
                t_exe_in = [t_exe_in t_exe];
                x(:,j) = [sj.Pgj sj.dgj sj.Pbj sj.Pplj sj.dplj sj.hbj sj.hplj]';
            end
    
            somma = zeros(2*M,1);
            for i=1:N
                somma = somma + Ai*x(:,i);
            end
    
            v(:,ki+1) = somma - b;
            a = alpha1/((ki)^alpha2);
            lambda_1 = lambda(:,ki);
            if (ki>1)%%||(ko>1) credo debba essere sempre dopo ki>1 perché v(:,1) non cambia mai
                lambda(:, ki+1) = max(lambda(:, ki) + a*(v(:, ki)+rho(:, ko)), zeros(2*M, 1));
                deltalambda = lambda(:, ki+1) - lambda(:, ki);
            else
                lambda(:, ki+1) = lambda(:,ki);
                deltalambda = 1;
            end

            disp("deltalambda=");
            max(abs(deltalambda))
            disp("ki ko");
            [ki ko]
            disp("max(v(:, ki))");
            max(v(:, ki))
    
            % Check convergence
            checkresult = [];
            if ki > 8
                for i = 1:7
                    deltalambda_c = lambda(:, ki-i+2) - lambda(:, ki-i+1);
                    lambda_c = lambda(:, ki-i+1);
                    relcon = max(deltalambda_c./lambda_c);
                    abscon = max(deltalambda_c);
    
                    if (relcon < 1e-3)||(abscon < 1e-4)
                        checkresult = [checkresult 1];
                    else
                        checkresult = [checkresult 0];
                    end
                end
    
                if checkresult == ones(1,7)
                    convergence = 1;
                end
            end

            ki = ki + 1;
            ki2 = ki2 +1;
            
    
        end
        ki_S = ki;
        for k = ki-w+1:ki %find ki_S
            if (k > 0) && (max(max(v(:,k),zeros(2*M,1))) < max(max(v(:,ki_S),zeros(2*M,1))))
                ki_S = k;
            end
        end
        rho(:, ko+1) = rho(:, ko) + max(v(:, ki_S), zeros(2*M,1));
    
        for t=1:M
            if rho(t, ko+1) + rho(t+M, ko+1) > 2*epsilon*gamma1
                    rho(t, ko+1) = rho(t, ko);
                    rho(t+M, ko+1) = rho(t+M, ko);
                    disp("NOT OK FAL");
                    out = 1;
            end
        end
        lambda(:, 1) = lambda(:, ki-1);
        ko = ko+1;
        disp("max(v(:, ki))");
        max(v(:, ki_S))
    
        maxv_show_next = max(v(:, ki));

    end
            
    if out == 0

        deltalambda_c = lambda(:, ki) - lambda(:, ki-1);
        lambda_c = lambda(:, ki);
        relcon = max(deltalambda_c./lambda_c);
        abscon = max(deltalambda_c);
        relcon_arr = [relcon_arr relcon];
        abscon_arr = [abscon_arr abscon];
    
        ki_arr = [ki_arr ki];
        ko_arr = [ko_arr ko];
        lambda_arr = [lambda_arr norm(lambda)];
        rho_arr = [rho_arr norm(rho)];
        max_v_arr = [max_v_arr max(v(:, ki))];
        epsilon_arr = [epsilon_arr epsilon];
        dlambda_arr = [dlambda_arr max(abs(deltalambda))];
        %Plot current advance, lambda and rho values
        subplot(3, 3, 1)
        yyaxis left
        plot(ki_arr,"-o",'LineWidth',1.5)
        yyaxis right
        plot(ko_arr,"-^",'LineWidth',1.5)
        title("k_i and k_o")
        legend("k_i","k_o",'Location','southeast')
        
        subplot(3, 3, 2)
        title("|\lambda| and |\rho|")
        yyaxis left
        lmax = max(lambda_arr);
        lmin = min(lambda_arr);
        ylim([lmin*0.9 lmax*1.1])
        plot(lambda_arr,"-o",'LineWidth',1.5)
        yyaxis right
        rmax = max(0.0001, max(rho_arr));
        rmin = min(rho_arr);
        ylim([rmin*0.9 rmax*1.1])
        xlim([0 length(rho_arr)])
        plot(rho_arr,"-^",'LineWidth',1.5)
        legend("|\lambda|","|\rho|",'Location','southeast')
    
        subplot(3, 3, 3)
        hold on
        yyaxis left
        plot(dlambda_arr,"-^",'LineWidth',1.5)
        plot(ones(length(dlambda_arr))*0.001,"--g",'LineWidth',1.5)
        dlmax = max(dlambda_arr);
        ylim([0.00085 dlmax*1.1])
        xlim([0 length(dlambda_arr)])
        set(gca, 'YScale', 'log')
        yyaxis right
        emax = max(0.0001, max(epsilon_arr));
        emin = min(epsilon_arr);
        ylim([emin*0.9 emax*1.1])
        xlim([0 length(epsilon_arr)])
        plot(epsilon_arr,"-^",'LineWidth',1.5)
        title("\Delta\lambda Max and \epsilon")
        legend("\Delta\lambda Max","\epsilon",'Location','southeast')
        hold off
        
        subplot(3, 3, [4, 5, 7, 8])
        hold on
        plot(max_v_arr,"-^b",'LineWidth',1.5)
        plot(zeros(length(max_v_arr)),"--g",'LineWidth',1.5)
        hold off
        vmax = max(max_v_arr);
        ylim([-vmax*0.15 vmax*1.1])
        xlim([0 length(max_v_arr)])
        title("Max Violation")
        
        subplot(3, 3, 9)
        hold on
        plot(abscon_arr,"-^",'LineWidth',1.5)
        plot(ones(length(abscon_arr))*1e-5,"--g",'LineWidth',1.5)
        set(gca, 'YScale', 'log')
        title("Absolute Convergence")
        acmax = max(abscon_arr);
        acmin = min(abscon_arr);
        ylim([0.1*acmin acmax*1.1])
        xlim([0 length(epsilon_arr)])
        hold off
        
        subplot(3, 3, 6)
        hold on
        plot(relcon_arr,"-^",'LineWidth',1.5)
        plot(ones(length(abscon_arr))*1e-3,"--g",'LineWidth',1.5)
        set(gca, 'YScale', 'log')
        title("Relative Convergence")
        rcmax = max(relcon_arr);
        rcmin = min(relcon_arr);
        ylim([0.1*rcmin rcmax*1.1])
        xlim([0 length(epsilon_arr)])
        hold off
        drawnow
        savefig(algo_str+"_"+N+"_SolutionProgress_epsilon_"+epsilon+".fig")
    
    
        
        previousgraph = [];
        figure
        sgtitle("Power profiles " + N + " prosumers" + "\n" + algo_str)
        for i=1:3
            rng('shuffle');
            pros = randi(N);
            while ismember(pros, previousgraph) ~= 0
                pros = randi(N);
            end
            previousgraph = [previousgraph pros];
            Pg = x(1:M,pros);
            Pb = x(M*2+1:M*3,pros);
            Ppl = x(M*3+1:M*4,pros);
        subplot(3,3,i)
        hold on
        plot(Pg);
        plot(Pg_ref(pros,:)'); 
        if i == 1
            ylabel({"Generator Power","",'P_g'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_g'+ "_" + pros)
        end
        xlabel("time")
        title("Prosumer " + pros)
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+3)
        hold on
        plot(Pb);
        plot(Pb_ref(pros,:)');
        if i == 1
            ylabel({"Battery Power","",'P_b'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_b'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+6)
        hold on
        plot(Ppl);
        plot(Ppl_ref(pros,:)');
        if i == 1
            ylabel({"Programmable Load Power","",'P_{pl}'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_p_l'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
        
        end
        drawnow
        savefig(algo_str+"_"+N+"_prosumer_ProsumerProfile"+".fig")

        figure
        title(algo_str + ". " + "Total Profile with " + N + " prosumers")
        hold on
        plot(sum(x(1:M,:)'+x(M*2+1:M*3,:)'-x(M*3+1:M*4,:)',1));
        title("Total Profile with " + N + " prosumers")
        subtitle(algo_str)
        plot(sum(P_ref,1));
        lightGreen = [0.85, 1, 0.85];
        patch([1:M fliplr(1:M)], [(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1) fliplr(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1)], lightGreen); 
        plot(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1);
        plot(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1);
        ylabel('P')
        xlabel("time")
        legend("Solution","Reference profile","TSO Request Admissible Region","Reduced Lower Bound","Reduced Upper Bound",'Location','southeast')
        hold off
        drawnow
        savefig(algo_str+"_"+N+"_pool_ProsumerProfile"+".fig")
    else
        close
    end

    costFal = sum(sum(Cg.*x(1:M,:)' + Cb.*x(M*5+1:M*6,:)' + Cpl.*x(M*6+1:M*7,:)',2),1);
    x_opt = x;
    fval = sum(sum(Cg.*x(1:M,:)' + Cb.*x(M*5+1:M*6,:)' + Cpl.*x(M*6+1:M*7,:)',2),1);
    ki_tot = ki2;
    ko = ko;
    epsilon_f = epsilon;
    t_exe_in = t_exe_in;
end
%% Distributed algorithm (Vujanic et al)

function [x_opt, fval, ki_tot, ko, epsilon_v, t_exe_in, out] = decentralized_vuj(N, M, maxPpl, Epl, maxPb, S0, minS, maxS, minPg, maxPg, Pg_ref, Pb_ref, Ppl_ref, P_ref, dtso, gamma1, epsilon, Cg, Cb, Cpl, alpha1_n, alpha2, taus, tr1, tf1)
    
    
    
    % DEFINE NEW COST MATRIX
    c_T = [Cg zeros(N,M) zeros(N,M) zeros(N,M) zeros(N,M) Cb Cpl];
    algo_str = "Vujanic et al";
    figtitle = algo_str + " with " + N + " prosumers" + "(\epsilon = " + epsilon + ")";
    figure('Name',figtitle);
    

    % DEFINE Ai MATRIX AND b VECTOR
    Ai1 = zeros(M,7*M);
    Ai2 = zeros(M,7*M);
    Mdtso1 = [];
    Mdtso2 = [];
    Mdtso_1 = [];
    Mdtso_2 = [];
    t_exe_in = [];
    
    for i=1:M
        Mdtso1 = [Mdtso1; dtso];
        Mdtso2 = [Mdtso2  dtso'];
    end
    for i = 1:7
        Mdtso_1 = [Mdtso_1 Mdtso1];
        Mdtso_2 = [Mdtso_2 Mdtso2];
    end
    
    for i=1:M
        for j=1:M
            if (i == j)
                Ai1(i,j) = -1;
                Ai1(i,M+j) = 0;
                Ai1(i,2*M+j) = -1;
                Ai1(i,3*M+j) = 1;
                Ai1(i,4*M+j) = 0;
                Ai1(i,5*M+j) = 0;
                Ai1(i,6*M+j) = 0;
                Ai2(i,j) = 1;
                Ai2(i,M+j) = 0;
                Ai2(i,2*M+j) = 1;
                Ai2(i,3*M+j) = -1;
                Ai2(i,4*M+j) = 0;
                Ai2(i,5*M+j) = 0;
                Ai2(i,6*M+j) = 0;
            end
        end
    end
    
    Ai=[Ai1.*Mdtso_1.*Mdtso_2;Ai2.*Mdtso_1.*Mdtso_2];
    b = [((-(1-epsilon*sign(gamma1))*gamma1-sum(P_ref,1)).*dtso)'
         ((+(1+epsilon*sign(gamma1))*gamma1+sum(P_ref,1)).*dtso)'];

    % ALGORITHM
    ki_arr = [];
    ko_arr = [];
    lambda_arr = [];
    dlambda_arr = [];
    rho_arr = [];
    max_v_arr = [];
    epsilon_arr = [];
    relcon_arr = [];
    abscon_arr = [];

    alpha1 = alpha1_n/N;
    
    lambda(:, 1) = zeros(2*M, 1);
    rho(:, 1) = zeros(2*M, 1);
    v(:, 1) = 10000*ones(2*M, 1);
    ko = 1;
    deltalambda = 1000;
    
    ki=1;
    ki2 = 0;
    out = 0;

    while max(v(:, ki)>0) && ko<15 && out==0
        ki = 1;
        deltalambda = 1;
        convergence = 0;
        while (convergence == 0) && (max(v(:, ki)>0))   % || a >=0.0002
    
            %PLOT FROM PREVIOUS RUN
            if (ki>1)||(ko>1) %skip first step

                if ki > 1
                    deltalambda_c = lambda(:, ki) - lambda(:, ki-1);
                    lambda_c = lambda(:, ki-1);
                    relcon = max(deltalambda_c./lambda_c);
                    abscon = max(deltalambda_c);
                end
                relcon_arr = [relcon_arr relcon];
                abscon_arr = [abscon_arr abscon];
    
                ki_arr = [ki_arr ki];
                ko_arr = [ko_arr ko];
                lambda_arr = [lambda_arr norm(lambda)];
                rho_arr = [rho_arr norm(rho)];
                if ki > 1
                    max_v_arr = [max_v_arr max(v(:, ki))];
                else
                    max_v_arr = [max_v_arr maxv_show_next];
                end
                epsilon_arr = [epsilon_arr epsilon];
                dlambda_arr = [dlambda_arr max(abs(deltalambda))];
                %Plot current advance, lambda and rho values
                subplot(3, 3, 1)
                yyaxis left
                plot(ki_arr,"-o",'LineWidth',1.5)
                yyaxis right
                hold on
                plot(ko_arr,"-^",'LineWidth',1.5)
                hold off
                title("k_i and k_o")
                legend("k_i","k_o",'Location','southeast')
                
                subplot(3, 3, 2)
                title("|\lambda| and |\rho|")
                yyaxis left
                lmax = max(lambda_arr);
                lmin = min(lambda_arr);
                if (ki>2)||(ko>1)
                    ylim([lmin*0.9 lmax*1.1])
                end
                plot(lambda_arr,"-o",'LineWidth',1.5)
                yyaxis right
                rmax = max(0.0001, max(rho_arr));
                rmin = min(rho_arr);
                ylim([rmin*0.9 rmax*1.1])
                xlim([0 length(rho_arr)])
                plot(rho_arr,"-^",'LineWidth',1.5)
                legend("|\lambda|","|\rho|",'Location','southeast')
    
                subplot(3, 3, 3)
                hold on
                plot(dlambda_arr,"-^",'LineWidth',1.5)
                dlmax = max(dlambda_arr);
                dlmin = min(dlambda_arr);
                ylim([0.9*dlmin dlmax*1.1])
                xlim([0 length(dlambda_arr)])
                set(gca, 'YScale', 'log')
                title("\Delta\lambda Max")
                hold off
    
                if ki > 1
                    subplot(3, 3, [4, 5, 7, 8])
                    hold on
                    plot(max_v_arr,"-^b",'LineWidth',1.5)
                    plot(zeros(length(max_v_arr)),"--g",'LineWidth',1.5)
                    hold off
                    vmax = max(max_v_arr);
                    ylim([-vmax*0.15 vmax*1.1])
                    xlim([0 length(max_v_arr)])
                    title("Max Violation")
                end
                
                subplot(3, 3, 9)
                hold on
                plot(abscon_arr,"-^",'LineWidth',1.5)
                plot(ones(length(abscon_arr))*1e-4,"--g",'LineWidth',1.5)
                set(gca, 'YScale', 'log')
                title("Absolute Convergence")
                acmax = max(abscon_arr);
                acmin = min(abscon_arr);
                if (ki>2)||(ko>1)
                    ylim([0.1*acmin acmax*1.1])
                end
                xlim([0 length(epsilon_arr)])
                hold off
                
                subplot(3, 3, 6)
                hold on
                plot(relcon_arr,"-^",'LineWidth',1.5)
                plot(ones(length(abscon_arr))*1e-3,"--g",'LineWidth',1.5)
                set(gca, 'YScale', 'log')
                title("Relative Convergence")
                rcmax = max(relcon_arr);
                rcmin = min(relcon_arr);
                if (ki>3)||(ko>1)
                    ylim([0.1*rcmin rcmax*1.1])
                end
                xlim([0 length(epsilon_arr)])
                hold off
                drawnow
            end
    
            parfor j=1:N   %find local optimum
                tic
                probj = optimproblem;
                % Programmable Load (variables definition and constraints)
                Pplj = optimvar('Pplj',1, M,'LowerBound',0); %programmable load power
                dplj = optimvar('dplj',1, M,'Type','integer','LowerBound',0,'UpperBound',4);
                probj.Constraints.conspl1 = Pplj == (dplj/4).*maxPpl(j,:);
                probj.Constraints.conspl2 = Epl(j,:) == sum(taus*Pplj,2);
                % Battery (variables definition and constraints)
                Pbj = optimvar('Pbj',1, M); %battery power
                minPb = -maxPb;
                consbj1 = minPb(j,:) <= Pbj;
                consbj2 = Pbj <= maxPb(j,:);
                consbj3 = minPb(j,:) <= Pbj; %random initialization
                consbj4 = Pbj <= maxPb(j,:); %random initialization
                for t=1:M
                    consbj3(:,t) = minS(j,:) <= S0(j,:) - taus*sum(Pbj(:,1:t),2);
                    consbj4(:,t) = S0(j,:) - taus*sum(Pbj(:,1:t),2) <= maxS(j,:);    
                end
                probj.Constraints.consb1 = consbj1;
                probj.Constraints.consb2 = consbj2;
                probj.Constraints.consb1 = consbj3;
                probj.Constraints.consb2 = consbj4;
                % Generator (variables definition and constraints)
                Pgj = optimvar('Pgj',1, M,'LowerBound',0); %generator power
                dgj = optimvar('dgj',1, M,'Type','integer','LowerBound',0,'UpperBound',1); %on-off state
                consgj1 = dgj.*minPg(j,:) <= Pgj;
                consgj2 = Pgj <= dgj.*maxPg(j,:);
                probj.Constraints.consg1 = consgj1;
                probj.Constraints.consg2 = consgj2;
                % Reference profile constraints
                probj.Constraints.cons1 = Pgj(:, 1:tr1) == Pg_ref(j, 1:tr1);
                probj.Constraints.cons2 = Pbj(:, 1:tr1) == Pb_ref(j, 1:tr1);   %you need to keep everything as it was since
                probj.Constraints.cons3 = Pplj(:, 1:tr1) == Ppl_ref(j, 1:tr1); %before tr1 you don't even know the request
                probj.Constraints.cons4 = Pgj(:, tf1+1:M) + Pbj(:, tf1+1:M) - Pplj(:, tf1+1:M) == P_ref(j, tf1+1:M);
                % Auxiliary constraints
                hbj = optimvar('hbj',1, M);   %auxiliary variable to make the cost function linear
                hplj = optimvar('hplj',1, M); %auxiliary variable to make the cost function linear
                
                consAuxj1 = minPb(j,:) <= Pbj;
                consAuxj2 = minPb(j,:) <= Pbj;
                consAuxj1(:,1) = zeros(1,1) <= hbj(:,1);
                consAuxj2(:,1) = zeros(1,1) <= hbj(:,1);
                for t=2:M  %at time 1 we don't have memory of the past
                    consAuxj1(:,t) = Pbj(:,t) - Pbj(:,t-1) <= hbj(:,t);
                    consAuxj2(:,t) = -Pbj(:,t) + Pbj(:,t-1) <= hbj(:,t);
                end
                consAuxj3 = Pplj - Ppl_ref(j,:) <= hplj;
                consAuxj4 = -Pplj + Ppl_ref(j,:) <= hplj;
                probj.Constraints.consAux1 = consAuxj1;
                probj.Constraints.consAux2 = consAuxj2;
                probj.Constraints.consAux3 = consAuxj3;
                probj.Constraints.consAux4 = consAuxj4;
    
                % Objective Function (costs, auxiliary variables and additional constraints)
                %costj = sum(Cg(j,:).*Pgj + Cb(j,:).*hbj + Cpl(j,:).*hplj,2);
                costj = c_T(j,:) + lambda(:, ki)'*Ai;
                xj = [Pgj dgj Pbj Pplj dplj hbj hplj]';
                probj.Objective = costj*xj;
    
    
                sj = solve(probj);
                t_exe = toc;
                t_exe_in = [t_exe_in t_exe];
                x(:,j) = [sj.Pgj sj.dgj sj.Pbj sj.Pplj sj.dplj sj.hbj sj.hplj]';
            end
            
            somma = zeros(2*M,1);
            for i=1:N
                somma = somma + Ai*x(:,i);
            end
            
            v(:,ki+1) = somma - b;
            a = alpha1/((ki)^alpha2);
            lambda_1 = lambda(:,ki);
            if (ki>1)%%||(ko>1) credo debba essere sempre dopo ki>1 perché v(:,1) non cambia mai
                lambda(:, ki+1) = max(lambda(:, ki) + a*(v(:, ki)+rho(:, ko)), zeros(2*M, 1));
                deltalambda = lambda(:, ki+1) - lambda(:, ki);
            else
                lambda(:, ki+1) = lambda(:,ki);
                deltalambda = 1;
            end
    %                 lambda(:, ki+1) = max(lambda(:, ki) + a*(v(:, ki)+rho(:, ko)), zeros(2*M, 1));
    %                 deltalambda = lambda(:, ki+1) - lambda(:, ki);
    
            ki = ki + 1;
            ki2 = ki2 +1;
            disp("deltalambda=");
            max(abs(deltalambda))
            disp("ki ko");
            [ki ko]
            disp("max(v(:, ki))");
            max(v(:, ki))
    
            % Check convergence
            checkresult = [];
            if ki > 8
                for i = 1:7
                    deltalambda_c = lambda(:, ki-i+1) - lambda(:, ki-i);
                    lambda_c = lambda(:, ki-i+1);
                    relcon = max(deltalambda_c./lambda_c);
                    abscon = max(deltalambda_c);
    
                    if (relcon < 1e-3)||(abscon < 1e-5)
                        checkresult = [checkresult 1];
                    else
                        checkresult = [checkresult 0];
                    end
                end
    
                if checkresult == ones(1,7)
                    convergence = 1;
                end
            end
        end
        
        si_max(:, 1) = -10000000*ones(2*M,1); % altro algoritmo (vujanic et al.)
        si_min(:, 1) = 10000000*ones(2*M,1);
        rho(:, ko+1) = zeros(2*M, 1);
        for i=1:N
            si_max(:,ko+1) = max(si_max(:,ko) , Ai*x(:,i));
            si_min(:,ko+1) = min(si_min(:,ko) , Ai*x(:,i));
            rhoi(:, i) = si_max(:,ko+1) - si_min(:,ko+1);
            for j=1:2*M
                if rho(j, ko+1) < rhoi(j, i)
                    rho(j, ko+1) = rhoi(j, i);
                end
            end
        end
    %             rho(:, ko+1) = 2*M*rho(:, ko+1); %oppure
        rho(:, ko+1) = 2*(nnz(dtso))*rho(:, ko+1);
        for t=1:M
            if rho(t, ko+1) + rho(t+M, ko+1) > 2*epsilon*gamma1
                    rho(t, ko+1) = rho(t, ko);
                    rho(t+M, ko+1) = rho(t+M, ko);
                    disp("NOT OK VUJ");
                    out = 1;
            end
        end
        lambda(:, 1) = lambda(:, ki);
    
        maxv_show_next = max(v(:, ki));
        ko = ko+1;
    end
            
    if out == 0

        deltalambda_c = lambda(:, ki) - lambda(:, ki-1);
        lambda_c = lambda(:, ki);
        relcon = max(deltalambda_c./lambda_c);
        abscon = max(deltalambda_c);
        relcon_arr = [relcon_arr relcon];
        abscon_arr = [abscon_arr abscon];
    
        ki_arr = [ki_arr ki];
        ko_arr = [ko_arr ko];
        lambda_arr = [lambda_arr norm(lambda)];
        rho_arr = [rho_arr norm(rho)];
        max_v_arr = [max_v_arr max(v(:, ki))];
        epsilon_arr = [epsilon_arr epsilon];
        dlambda_arr = [dlambda_arr max(abs(deltalambda))];
        %Plot current advance, lambda and rho values
        subplot(3, 3, 1)
        yyaxis left
        plot(ki_arr,"-o",'LineWidth',1.5)
        yyaxis right
        plot(ko_arr,"-^",'LineWidth',1.5)
        title("k_i and k_o")
        legend("k_i","k_o",'Location','southeast')
        
        subplot(3, 3, 2)
        title("|\lambda| and |\rho|")
        yyaxis left
        lmax = max(lambda_arr);
        lmin = min(lambda_arr);
        ylim([lmin*0.9 lmax*1.1])
        plot(lambda_arr,"-o",'LineWidth',1.5)
        yyaxis right
        rmax = max(0.0001, max(rho_arr));
        rmin = min(rho_arr);
        ylim([rmin*0.9 rmax*1.1])
        xlim([0 length(rho_arr)])
        plot(rho_arr,"-^",'LineWidth',1.5)
        legend("|\lambda|","|\rho|",'Location','southeast')
    
        subplot(3, 3, 3)
        hold on
        yyaxis left
        plot(dlambda_arr,"-^",'LineWidth',1.5)
        plot(ones(length(dlambda_arr))*0.001,"--g",'LineWidth',1.5)
        dlmax = max(dlambda_arr);
        ylim([0.00085 dlmax*1.1])
        xlim([0 length(dlambda_arr)])
        set(gca, 'YScale', 'log')
        yyaxis right
        emax = max(0.0001, max(epsilon_arr));
        emin = min(epsilon_arr);
        ylim([emin*0.9 emax*1.1])
        xlim([0 length(epsilon_arr)])
        plot(epsilon_arr,"-^",'LineWidth',1.5)
        title("\Delta\lambda Max and \epsilon")
        legend("\Delta\lambda Max","\epsilon",'Location','southeast')
        hold off
        
        subplot(3, 3, [4, 5, 7, 8])
        hold on
        plot(max_v_arr,"-^b",'LineWidth',1.5)
        plot(zeros(length(max_v_arr)),"--g",'LineWidth',1.5)
        hold off
        vmax = max(max_v_arr);
        ylim([-vmax*0.15 vmax*1.1])
        xlim([0 length(max_v_arr)])
        title("Max Violation")
        
        subplot(3, 3, 9)
        hold on
        plot(abscon_arr,"-^",'LineWidth',1.5)
        plot(ones(length(abscon_arr))*1e-5,"--g",'LineWidth',1.5)
        set(gca, 'YScale', 'log')
        title("Absolute Convergence")
        acmax = max(abscon_arr);
        acmin = min(abscon_arr);
        ylim([0.1*acmin acmax*1.1])
        xlim([0 length(epsilon_arr)])
        hold off
        
        subplot(3, 3, 6)
        hold on
        plot(relcon_arr,"-^",'LineWidth',1.5)
        plot(ones(length(abscon_arr))*1e-3,"--g",'LineWidth',1.5)
        set(gca, 'YScale', 'log')
        title("Relative Convergence")
        rcmax = max(relcon_arr);
        rcmin = min(relcon_arr);
        ylim([0.1*rcmin rcmax*1.1])
        xlim([0 length(epsilon_arr)])
        hold off
        drawnow
        savefig(algo_str+"_"+N+"_SolutionProgress_epsilon_"+epsilon+".fig")
    
    
        
        previousgraph = [];
        figure
        sgtitle("Power profiles " + N + " prosumers" + "\n" + algo_str)
        for i=1:3
            rng('shuffle');
            pros = randi(N);
            while ismember(pros, previousgraph) ~= 0
                pros = randi(N);
            end
            previousgraph = [previousgraph pros];
            Pg = x(1:M,pros);
            Pb = x(M*2+1:M*3,pros);
            Ppl = x(M*3+1:M*4,pros);
        subplot(3,3,i)
        hold on
        plot(Pg);
        plot(Pg_ref(pros,:)'); 
        if i == 1
            ylabel({"Generator Power","",'P_g'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_g'+ "_" + pros)
        end
        xlabel("time")
        title("Prosumer " + pros)
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+3)
        hold on
        plot(Pb);
        plot(Pb_ref(pros,:)');
        if i == 1
            ylabel({"Battery Power","",'P_b'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_b'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
    
        subplot(3,3,i+6)
        hold on
        plot(Ppl);
        plot(Ppl_ref(pros,:)');
        if i == 1
            ylabel({"Programmable Load Power","",'P_{pl}'+ "_" + pros},'interpreter','latex')
        else
            ylabel('P_p_l'+ "_" + pros)
        end
        xlabel("time")
        legend("Solution","Reference profile",'Location','southeast')
        hold off
        drawnow
        
        end
        drawnow
        savefig(algo_str+"_"+N+"_prosumer_ProsumerProfile"+".fig")

        figure
        title(algo_str + ". " + "Total Profile with " + N + " prosumers")
        hold on
        plot(sum(x(1:M,:)'+x(M*2+1:M*3,:)'-x(M*3+1:M*4,:)',1));
        title("Total Profile with " + N + " prosumers")
        subtitle(algo_str)
        plot(sum(P_ref,1));
        lightGreen = [0.85, 1, 0.85];
        patch([1:M fliplr(1:M)], [(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1) fliplr(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1)], lightGreen); 
        plot(sum(P_ref,1) + (1-epsilon*sign(gamma1))*gamma1-rho(1:96,ko)');
        plot(sum(P_ref,1) + (1+epsilon*sign(gamma1))*gamma1-rho(97:192,ko)');
        ylabel('P')
        xlabel("time")
        legend("Solution","Reference profile","TSO Request Admissible Region","Tightened Lower Bound","Tightened Upper Bound",'Location','southeast')
        hold off
        drawnow
        savefig(algo_str+"_"+N+"_pool_ProsumerProfile"+".fig")
    else
        close
    end

    x_opt = x;
    fval = sum(sum(Cg.*x(1:M,:)' + Cb.*x(M*5+1:M*6,:)' + Cpl.*x(M*6+1:M*7,:)',2),1);
    ki_tot = ki2;
    ko = ko;
    epsilon_v = epsilon;
    t_exe_in = t_exe_in;
end
