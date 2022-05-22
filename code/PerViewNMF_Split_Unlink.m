function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = PerViewNMF_Split_Unlink(X, k, Vo, options, U, V, weight)
% This is a module of Multi-View Non-negative Matrix Factorization
% (MultiNMF) for the update for one view as in lines 5-9 in Alg. 1
%
% Notation:
% X ... (mFea x nSmp) data matrix of one view
%       mFea  ... number of features
%       nSmp  ... number of samples
% k ... number of hidden factors
% Vo... consunsus
% options ... Structure holding all settings
% U ... initialization for basis matrix 
% V ... initialization for coefficient matrix 
%
%   Originally written by Deng Cai (dengcai AT gmail.com) for GNMF
%   Modified by Jialu Liu (jliu64@illinois.edu)

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIterOrig = options.minIter;
minIter = minIterOrig-1;
meanFitRatio = options.meanFitRatio;

alpha = weight;

Norm = 1;
NormV = 0;

[mFea,nSmp]=size(X);

bSuccess.bSuccess = 1;

selectInit = 1;
if isempty(U)
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end

[U,V] = Normalize(U, V);
if nRepeat == 1
    selectInit = 0;
    minIterOrig = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, Vo, alpha);
        meanFit = objhistory*10;
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, Vo, alpha);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
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
        XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU = V*UU; % nk^2
        
        XU = XU + alpha * Vo;
        VUU = VUU + alpha * V;
        
        V = V.*(XU./max(VUU,1e-10));
    
        % ===================== update U ========================
        XV = X*V; 
        VV = V'*V;
        UVV = U*VV;
        
        VV_ = repmat(diag(VV)' .* sum(U, 1), mFea, 1);
        tmp = sum(V.*Vo);
        VVo = repmat(tmp, mFea, 1);
        
        XV = XV + alpha * VVo;
        UVV = UVV + alpha * VV_;

        U = U.*(XV./max(UVV,1e-10)); 
        
        [U,V] = Normalize(U, V);
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, Vo, alpha);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, Vo, alpha);
                    objhistory = [objhistory newobj]; 
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, Vo, alpha);
                        objhistory = [objhistory newobj]; 
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
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
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
        bSuccess.nStepTrial = nStepTrial;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
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
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            [U,V] = Normalize(U, V);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
            
        end
    end
end

nIter_final = nIter_final + minIterOrig;
[U_final, V_final] = Normalize(U_final, V_final);