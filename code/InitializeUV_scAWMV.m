function [U_ini, V_ini] = InitializeUV_scAWMV(data, K, options)
%label = gnd;
% Written by Jialu Liu (jliu64@illinois.edu)
% Modified by Pengcheng Zeng (pchzengncl@gmail.com)
U_ini = cell(1, length(data));
V_ini = cell(1, length(data));
viewNum = length(data);
U_ = [];
V_ = [];
j = 0;
while j < 3
    j = j + 1;
    if j == 1
        [U_ini{1}, V_ini{1}] = NMF(data{1}, K, options, U_, V_);
    else
        [U_ini{1}, V_ini{1}] = NMF(data{1}, K, options, U_, V_ini{viewNum});     
    end
    for i = 2:viewNum
        [U_ini{i}, V_ini{i}] = NMF(data{i}, K, options, U_, V_ini{i-1});
    end
end