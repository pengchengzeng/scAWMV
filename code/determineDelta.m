function [TEMP, gamma0] = determineDelta(X, U_ini, V_ini)
%X = data; label = gnd; 
viewNum = length(X);

U = U_ini;
V = V_ini;

TEMP = zeros(viewNum,1); 

% initialize view weight; by PC
%weight = view_weights(X, U, V, delta, viewNum);
loggL=0;
for i = 1:viewNum
   tmp1 = X{i} - U{i}*V{i}';
   TEMP(i) = sum(sum(tmp1.^2))/log10(1/viewNum);
   %loggL = loggL + weight(i) * sum(sum(tmp1.^2)) + alpha(i) * sum(sum(tmp2.^2))+ delta * weight(i) * log10(weight(i)); %by PC
   loggL = loggL + 1/viewNum * sum(sum(tmp1.^2));
end

tmp3 = U{1}-U{2};
gamma0 = loggL/sum(sum(tmp3.^2));
end