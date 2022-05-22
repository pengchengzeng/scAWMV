function [ac, indic] = printResult_pc(X, data_score, label, K, kmeansFlag)
%X=V_ini{i}; K=K0; kmeansFlag=options.kmeans;
%X = V_scAI.'; data_score= data_score; label= gnd;kmeansFlag=1;
if kmeansFlag == 1
    indic = litekmeans(X, K, 'Replicates',20);
else
    [~, indic] = max(X, [] ,2);
end
%result = bestMap(label, indic);
[ac, nmi_value, cnt] = CalcMetrics(label, indic);
[ari_value,~,~,~] = RandIndex(label,indic);
ragi = calculate_ragi_score(indic, data_score{1}, data_score{2});
disp(sprintf('ac: %0.4f\t%d/%d\tNMI:%0.4f\t\tARI:%0.4f\t\tRAGI:%0.4f\t', ac, cnt, length(label), nmi_value, ari_value, ragi));