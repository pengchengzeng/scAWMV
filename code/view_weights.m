function [weight0] = view_weights(X, U, V, delta, viewNum)
weight0 = zeros(1, viewNum);
wexp = zeros(1,viewNum);
for i = 1:viewNum
    tmp0 = X{i} - U{i}*V{i}';
   wexp(i) = exp(-sum(sum(tmp0.^2))/delta);
end
for i = 1:viewNum
    weight0(i) =  wexp(i)/sum(wexp);
end
end