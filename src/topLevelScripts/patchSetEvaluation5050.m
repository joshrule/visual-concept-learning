home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

imgDir = [home 'imageSets/imageNet/'];
outDir = ensureDir([home 'evaluation/5050v4/']); % change after commit
organicImgDir   = [imgDir   'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];
organicC3Dir   = [home   'patchSets/organicC3FullNegv2/']; % change after commit
inorganicC3Dir = [home 'patchSets/inorganicC3FullNegv2/']; % change after commit

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
nCategories = 50;

fprintf('\n');
fprintf('Initialized\n\n');

rng(0,'twister');
fprintf('Pseudorandom Number Generator Reset\nrng(0,''twister'')\n\n');

% k-means 400
if ~exist([outDir 'kmeans-5050-evaluation.mat'],'file')
    [c2,labels,organicC2Files,inorganicC2Files] = ...
      build5050C2(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir, ...
        'kmeans',nCategories);
    save([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
    fprintf('Built 5050 C2\n\n');

    if ~exist([outDir 'splits-5050.mat'],'file')
        nTrainingExamples = [16 32 64 128 256];
        nRuns = 20;
        cvsplit = cv(labels,nTrainingExamples,nRuns);
        save([outDir 'splits-5050.mat'],'nTrainingExamples','nRuns','cvsplit');
        fprintf('50/50 splits generated\n\n');
    else
        load([outDir 'splits-5050.mat'])
        fprintf('50/50 splits loaded\n\n');
    end

    [aucs,dprimes,models,classVals] = evaluatePerformance(c2,labels,cvsplit,method, ...
      options,size(c2,1),[]);
    save([outDir 'kmeans-5050-evaluation.mat'],'labels','c2','aucs', ...
      'dprimes', 'models', 'classVals', '-v7.3');
    clear animals noAnimals c2 labels aucs dprimes models classVals;
else
    load([outDir 'splits-5050.mat'],'cvsplit');
    load([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
end
fprintf('kmeans 50/50 evaluated\n\n');

c2Files = {organicC2Files{:} inorganicC2Files{:}};
c3Files = regexprep(c2Files,'c2','c3');

% cache organic C3
organicC3Files = regexprep(c3Files,'kmeans','organicOldSchoolFullNegv2'); % change after commit
load([organicC3Dir 'models.mat'],'models');
for i = 1:length(organicC3Files)
    if ~exist(organicC3Files{i},'file')
        cacheC3(organicC3Files{i},c2Files{i}, ...
          [organicC3Dir 'models.mat'],[organicC3Dir 'params.mat'],models);
        fprintf('%d: cached %s\n',i,organicC3Files{i});
    else
        fprintf('%d: found %s\n',i,organicC3Files{i});
    end
end

% cache inorganic C3
inorganicC3Files = regexprep(c3Files,'kmeans','inorganicOldSchoolFullNegv2'); % change after commit
load([inorganicC3Dir 'models.mat'],'models');
for i = 1:length(inorganicC3Files)
    if ~exist(inorganicC3Files{i},'file')
        cacheC3(inorganicC3Files{i},c2Files{i}, ...
          [inorganicC3Dir 'models.mat'],[inorganicC3Dir 'params.mat'],models);
	fprintf('%d: cached %s\n',i,inorganicC3Files{i});
    else
	fprintf('%d: found %s\n',i,inorganicC3Files{i});
    end
end

c3Files = {organicC3Files{:} inorganicC3Files{:}};
save([outDir 'chosenC3Categories.mat'],'organicC3Files','inorganicC3Files');

% organic
if ~exist([outDir 'organic-evaluation.mat'],'file')
    [c3,labels] = build5050C3(organicC3Files);
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('organic evaluated\n');

% organic-super
if ~exist([outDir 'organic-super-evaluation.mat'],'file')
    [c3,labels] = build5050C3(organicC3Files);
    load([outDir 'kmeans-5050-evaluation.mat'],'c2');
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'organic-super-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('organic-super evaluated\n');

% inorganic
if ~exist([outDir 'inorganic-evaluation.mat'],'file')
    [c3,labels] = build5050C3(inorganicC3Files);
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('inorganic evaluated\n');

% inorganic-super
if ~exist([outDir 'inorganic-super-evaluation.mat'],'file')
    [c3,labels] = build5050C3(inorganicC3Files);
    load([outDir 'kmeans-5050-evaluation.mat'],'c2');
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'inorganic-super-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('inorganic-super evaluated\n');

% combined
if ~exist([outDir 'combined-evaluation.mat'],'file')
    [c3a,labels] = build5050C3(organicC3Files);
    [c3b,labels] = build5050C3(inorganicC3Files);
    c3 = [c3a; c3b];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'combined-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('combined evaluated\n');

% combined-super
if ~exist([outDir 'combined-super-evaluation.mat'],'file')
    [c3a,labels] = build5050C3(organicC3Files);
    [c3b,labels] = build5050C3(inorganicC3Files);
    load([outDir 'kmeans-5050-evaluation.mat'],'c2');
    c3 = [c3a; c3b; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'combined-super-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear animals noAnimals c3 aucs dprimes models classVals;
end
fprintf('combined-super evaluated\n');

% % semantic analysis
% 
% nPairs = 1000;
% N = 25;
% blockSize = [6 6];
% maxSize = 240;
% targetWnids = listImageNetCategories(c3Files);
% featureDir = ensureDir([outDir 'featureCorrelations/']);
% simFile = [home 'src/external/similarity.pl'];
% load([outDir 'kmeans-5050-evaluation.mat'],'c2');
% 
% organicDir = ensureDir([featureDir 'organic/']);
% [c3,labels,imgNames] = build5050C3(organicC3Files);
% imgNames = regexprep(imgNames,'joshrule','josh');
% load([organicC3Dir 'splits.mat'],'testFiles');
% organicWnids = listImageNetCategories(testFiles);
% outFile = [organicDir 'pairwiseCorrelations.mat'];
% pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
% categoricalFeatureCorrelations(labels,imgNames,c2,c3,organicWnids, ...
%   targetWnids,N,outDir,simFile,blockSize,maxSize);
% 
% inorganicDir = ensureDir([featureDir 'inorganic/']);
% [c3,labels,imgNames] = build5050C3(inorganicC3Files);
% imgNames = regexprep(imgNames,'joshrule','josh');
% load([inorganicC3Dir 'splits.mat'],'testFiles');
% inorganicWnids = listImageNetCategories(testFiles);
% outFile = [inorganicDir 'pairwiseCorrelations.mat'];
% pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
% categoricalFeatureCorrelations(labels,imgNames,c2,c3,inorganicWnids, ...
%   targetWnids,N,outDir,simFile,blockSize,maxSize);
