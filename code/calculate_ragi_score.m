function ragi = calculate_ragi_score(clu, pbmc_markerscore, pbmc_housekeepingscore)
marker_avg = [];
housekeeping_avg = [];
marker_score = pbmc_markerscore;
housekeeping_score = pbmc_housekeepingscore;
clu_num = unique(clu);

for i = 1:size(clu_num,1)
    len = size(find(clu==clu_num(i)),1);
    if len > 1
        marker_avg = [marker_avg,mean(marker_score(:,clu==clu_num(i)),2)];
        housekeeping_avg = [housekeeping_avg,mean(housekeeping_score(:,clu==clu_num(i)),2)];
    else
        marker_avg = [marker_avg,marker_score(:,clu==clu_num(i))];
        housekeeping_avg = [housekeeping_avg,housekeeping_score(:,clu==clu_num(i))];
    end 
end
marker_gini_vec = zeros(0,size(marker_avg,1));
for j = 1: size(marker_avg,1)
        marker_gini_vec(j) = ginicoeff(marker_avg(j,:));
end
housekeeping_gini_vec = zeros(0,size(housekeeping_avg,1));
for j = 1: size(housekeeping_avg,1)
        housekeeping_gini_vec(j) = ginicoeff(housekeeping_avg(j,:));
end
marker_gini = mean(marker_gini_vec(~isnan(marker_gini_vec)));
housekeeping_gini = mean(housekeeping_gini_vec(~isnan(housekeeping_gini_vec)));
ragi = marker_gini - housekeeping_gini;
clear var marker_avg housekeeping_avg marker_score housekeeping_score 
