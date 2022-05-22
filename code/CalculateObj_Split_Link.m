%==========================================================================

function [obj, dV] = CalculateObj_Split_Link(X, U, U2, V, L, alpha, gamma, deltaVU, dVordU)
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    maxM = 62500000;
    [mFea, nSmp] = size(X);
    mn = numel(X);
    nBlock = floor(mn*3/maxM);

    if mn < maxM
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    tmp = V-L;
    obj_Lap = sum(sum(tmp.^2));
    
    tmp3 = U-U2;
    obj_Lap2 = sum(sum(tmp3.^2));
   
    dX = U*V'-X;
    obj_NMF = sum(sum(dX.^2));
    obj = obj_NMF+ alpha * obj_Lap + gamma * obj_Lap2;