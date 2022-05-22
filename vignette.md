# A quick guide to scAWMV method (Take example1A for example)

## 1. Load preprocessed data, RAGI socre matrix, and ground truth labels
```Matlab
load('./data/X11_example1A.mat'); load('./data/X12_example1A.mat'); load('./data/X21_example1A.mat'); load('./data/X22_example1A.mat'); 
load('./data/score_example1A.mat'); load('./data/gnd_example1A.mat');
data{1}=X11;data{2}=X21;data{3}=X12;data{4}=X22;
```

## 2. Parameter setting
```Matlab
K = 20;K0=9; % K: the number of factors; K0: the nunber of clusters.
options = [];
options.maxIter = 200;
options.error = 1e-6;
options.nRepeat = 30;
options.minIter = 50;
options.meanFitRatio = 0.1;
options.rounds = 35;
options.alpha = [0.01 0.01 0.01 0.01]; % options alpha is an array of weights for different views
options.kmeans = 1; % options.kmeans means whether to run kmeans on v^* or not
```


## 3. Initialize
```Matlab
rng(2022);
[U_ini, V_ini] = InitializeUV_scAWMV(data,K,options);
%printResult_pc(V_ini{1}, data_score,gnd, K0, options.kmeans);
[TEMP, gamma0] = determineDelta(data, U_ini, V_ini);
options.gamma = gamma0;
options.delta = -min(TEMP);
```

## 4. Run scAWMV algorithm
```Matlab
% Note that the clustering results shown in this step is based on k++means.
[U_final, V_final, V_centroid, ~, ~] = scAWMV_main(data, K, options, U_ini, V_ini);
printResult_pc(V_centroid, data_score,gnd, K0, options.kmeans);
```

## 5. Louvain clustering
Use the V_centroid as the input for Louvain clustering. See the python file Louvain_scAWMV.ipynb.
