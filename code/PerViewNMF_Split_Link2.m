function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = PerViewNMF_Split_Link2(X0, k0, Vo0, options0, U01, U02, V0, weight0, alpha0)
% This is a module of Multi-View Non-negative Matrix Factorization
% (MultiNMF) for the update for one view as in lines 5-9 in Alg. 1
%
% Notation:
% X0 ... (mFea x nSmp) data matrix of one view (linked view in one view)
%       mFea  ... number of features
%       nSmp  ... number of samples
% k0 ... number of hidden factors
% Vo0... consunsus
% options0 ... Structure holding all settings
% U01 ... initialization for basis matrix (linked view 1)
% U02 ... initialization for basis matrix (linked view 2)
% V0 ... initialization for coefficient matrix 
% weight0 ... the auto-weight (\omega) of one veiw
%
%   Originally written by Deng Cai (dengcai AT gmail.com) for GNMF
%   Modified by Jialu Liu (jliu64@illinois.edu)
%   Further modified by Pengcheng Zeng (pchzengncl@gmail.com)

%X0 = X{1}; k0 = K; Vo0 = centroidV; options0 = optionsForPerViewNMF; U01=U{1}; U02 = U{2}; V0 = V{1}; weight0 = weight(1); alpha0 = alpha(1);

differror = options0.error;
maxIter = options0.maxIter;
nRepeat = options0.nRepeat;
minIterOrig = options0.minIter;
minIter = minIterOrig-1;
meanFitRatio = options0.meanFitRatio;

alpha = alpha0/weight0;
gamma = options0.gamma/weight0;

Norm = 1;
NormV = 0;

[mFea,nSmp]=size(X0);

bSuccess.bSuccess = 1;

selectInit = 1;
if isempty(U01)
    U01 = abs(rand(mFea,k0));
    V0 = abs(rand(nSmp,k0));
else
    nRepeat = 1;
end

[U01,V0] = Normalize(U01, V0);
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj_Split_Link(X0, U01, U02, V0, Vo0, alpha, gamma);
        meanFit = objhistory*10;
    else
        if isfield(options0,'Converge') && options0.Converge
            objhistory = CalculateObj_Split_Link(X0, U01, U02, V0, Vo0, alpha, gamma);
        end
    end
else
    if isfield(options0,'Converge') && options0.Converge
        error('Not implemented!');
    end
end



tryNo = 0;
while tryNo < nRepeat   
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    nStepTrial = 0;
    %disp a
    while(maxErr > differror)
        % ===================== update V ========================
        XU = X0'*U01;  % mnk or pk (p<<mn)
        UU = U01'*U01;  % mk^2
        VUU = V0*UU; % nk^2
        
        XU = XU + alpha * Vo0;
        VUU = VUU + alpha * V0;
        
        V0 = V0.*(XU./max(VUU,1e-10));
    
        % ===================== update U ========================
        XV = X0*V0; 
        VV = V0'*V0;
        UVV = U01*VV;
        
        VV_ = repmat(diag(VV)' .* sum(U01, 1), mFea, 1);
        tmp = sum(V0.*Vo0);
        VVo = repmat(tmp, mFea, 1);
        
        XV = XV + alpha * VVo + gamma * U02; % modified by PC
        UVV = UVV + alpha * VV_ + gamma * U01; % modified by PC

        U01 = U01.*(XV./max(UVV,1e-10)); 
        
        [U01,V0] = Normalize(U01, V0);
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj_Split_Link(X0, U01, U02, V0, Vo0, alpha, gamma);
                % CalculateObj_Split_Link(X, U, U2, V, L, alpha, gamma, deltaVU, dVordU)
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj_Split_Link(X0, U01, U02, V0, Vo0, alpha, gamma);
                    objhistory = [objhistory newobj]; 
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options0,'Converge') && options0.Converge
                        newobj = CalculateObj_Split_Link(X0, U01, U02, V0, Vo0, alpha, gamma);
                        objhistory = [objhistory newobj]; 
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options0,'Converge') && options0.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    elapse = cputime - tmp_T;

    if tryNo == 1
        U_final = U01;
        V_final = V0;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
        bSuccess.nStepTrial = nStepTrial;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U01;
           V_final = V0;
           nIter_final = nIter;
           objhistory_final = objhistory;
           bSuccess.nStepTrial = nStepTrial;
           if selectInit
               elapse_final = elapse;
           else
               elapse_final = elapse_final+elapse;
           end
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U01 = abs(rand(mFea,k0));
            V0 = abs(rand(nSmp,k0));
            [U01,V0] = Normalize(U01, V0);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U01 = U_final;
            V0 = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
            
        end
    end
end

nIter_final = nIter_final + minIterOrig;
[U_final, V_final] = Normalize(U_final, V_final);



