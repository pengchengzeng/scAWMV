function [U, V, centroidV, logg, WEIGHT] = scAWMV_main(X, K, options, U_ini, V_ini)
% This is a module of Multi-View Non-negative Matrix
% Factorization(MultiNMF) by splitting both the linked and unlinked
% features in each data type with the following objective function: 
%\sum_{v=1}^{n_{v}}\omega^{(v)}(||X^{v}-U^{(v)}V^{(v)T}||^2 +
%\alpha||V^{(v)}-V*||^2)+ \gamma ||U_{link}^{(1)} -
%U_{link}^^{(2)}||^2 + \delta \sum_{v=1}^{n_{v}}\omega^{(v)} ln \omega^{(v)}.

% Notation:
% X ... a cell containing all views for the data
% K ... number of hidden factors
% K0... number of expected clusters
% label ... ground truth labels
% Written by Jialu Liu (jliu64@illinois.edu)
% Modified by Pengcheng Zeng (pchzengncl@gmail.com)

%X = data; label = gnd; 

viewNum = length(X);
Rounds = options.rounds;
alpha = options.alpha;
gamma = options.gamma;
delta = options.delta;


WEIGHT = zeros(viewNum,Rounds+1); %record the auto weights; by PC

U = U_ini;
V = V_ini;

logg = 0;

% initialize view weight; by PC
weight = view_weights(X, U, V, delta, viewNum);
WEIGHT(:,1) = weight;

optionsForPerViewNMF = options;
oldL = 100;

tic
j = 0;
while j < Rounds
    j = j + 1;
    if j==1
        centroidV = V{1};
    else
        centroidV = alpha(1) * V{1};
        for i = 2:viewNum
            centroidV = centroidV + alpha(i) * V{i};
        end
        centroidV = centroidV / sum(alpha);
    end
    loggL = 0;
    for i = 1:viewNum
        tmp1 = X{i} - U{i}*V{i}';
        tmp2 = V{i} - centroidV;
        loggL = loggL + weight(i) * sum(sum(tmp1.^2)) + alpha(i) * sum(sum(tmp2.^2))+ delta * weight(i) * log10(weight(i)); %by PC
    end
    
    %We let data{1}(corresponding to U{1}) and data{2}(corresponding to
    %U{2}) be the linked data from two data types (RNA and ATAC),
    %respectively.
    tmp3 = U{1}-U{2};
    loggL = loggL + gamma * sum(sum(tmp3.^2)); % add the L^2 penalty;
  
    logg(end+1) = loggL;
    if(oldL < loggL)
        U = oldU;
        V = oldV;
        loggL = oldL;
        j = j - 1;
        disp('restrart this iteration');
    end
    
    oldU = U;
    oldV = V;
    oldL = loggL;
    
    % Update the linked part, i.e. U{1}, V{1}, U{2} and V{2}.
    [U{1}, V{1}] = PerViewNMF_Split_Link2(X{1}, K, centroidV, optionsForPerViewNMF, U{1}, U{2}, V{1}, weight(1), alpha(1));
    [U{2}, V{2}] = PerViewNMF_Split_Link2(X{2}, K, centroidV, optionsForPerViewNMF, U{2}, U{1}, V{2}, weight(2), alpha(2));
    
    % Update the unlinked part
    %if length(X)>2
    for i = 3:4
        w = alpha(i)/weight(i);
        [U{i}, V{i}] = PerViewNMF_Split_Unlink(X{i}, K, centroidV, optionsForPerViewNMF, U{i}, V{i}, w);
    end
    %else
    % update view weight; by PC
    weight =  view_weights(X, U, V, delta, viewNum);
    WEIGHT(:,j+1) = weight; 
    %end
end
toc