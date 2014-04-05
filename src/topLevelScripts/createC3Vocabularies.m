home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

imgDir = [home 'imageSets/imageNet/'];
organicImgDir   = [imgDir 'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];
outDir = [home 'patchSets/'];

method = 'svm';
options = struct('svmTrainFlags', '-s 0 -t 0 -c 0.1 -b 1 -q', ...
                 'svmTestFlags', '-b 1', ...
                 'alpha', 0.7, ...
                 'startPerIter', 200, ...
                 'threshold', 0.25, ...
		 'ratio', 1.0);

load([outDir 'oldC3Categories.mat'],'organicFiles'); fprintf('files...');
[organicC2,organicLabels] = responsesFromCaches(organicFiles,'c2'); fprintf('c2...');
size(organicC2)
size(organicLabels)
models = trainC3(organicC2,organicLabels,method,options); fprintf('c3...');
save([outDir 'organicC3Old.isolated.mat'],'organicFiles','method','options','models','-v7.3');
clear models;
fprintf('saved...organic\n')

load([outDir 'oldC3Categories.mat'],'inorganicFiles'); fprintf('files...');
[inorganicC2,inorganicLabels] = responsesFromCaches(inorganicFiles,'c2'); fprintf('c2...');
models = trainC3(inorganicC2,inorganicLabels,method,options); fprintf('c3...');
save([outDir 'inorganicC3Old.isolated.mat'],'inorganicFiles','method','options','models','-v7.3');
clear models;
fprintf('saved...inorganic\n')

% create combined C3 classifiers
allFiles = [organicFiles; inorganicFiles];
fprintf('files...');
allC2 = [organicC2 inorganicC2];
allLabels = blkdiag(organicLabels,inorganicLabels);
fprintf('c2...');
models = trainC3(allC2,allLabels,method,options);
fprintf('c3...');
save([outDir 'organicC3Old.shared.mat'],'organicFiles','method','options','models','-v7.3');
save([outDir 'inorganicC3.shared.mat'],'inorganicFiles','method','options','models','-v7.3');
clear;
fprintf('saved...all\n')

% % % Old code to create category lists % % %
% create organic C3 classifiers
% organicFileData = dir([organicImgDir '*.kmeans.c2.mat']);
% organicNames = {organicFileData.name}';
% organicFiles = strcat(organicImgDir,organicNames(randperm(length(organicNames),1000)));

% create inorganic C3 classifiers
% inorganicFileData = dir([inorganicImgDir '*.kmeans.c2.mat']);
% inorganicNames = {inorganicFileData.name}';
% inorganicFiles = strcat(inorganicImgDir,inorganicNames(randperm(length(inorganicNames),1000)));
