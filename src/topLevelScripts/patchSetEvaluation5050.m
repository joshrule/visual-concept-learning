home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

imgDir = [home 'imageSets/imageNet/'];
outDir = ensureDir([home 'evaluation/5050v3.1/']); % change after commit
organicImgDir   = [imgDir   'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];
organicC3Dir   = [home   'patchSets/organicC3v2/']; % change after commit
inorganicC3Dir = [home 'patchSets/inorganicC3v2/']; % change after commit

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
nCategories = 50;

fprintf('\n');
fprintf('Initialized\n\n');

rng(0,'twister');
fprintf('Pseudorandom Number Generator Reset\nrng(0,''twister'')\n\n');

% k-means 400

if ~exist([outDir 'kmeans-5050-evaluation.mat'],'file')
    if ~exist([outDir 'chosenCategories.mat'],'file')
        [c2,labels,organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = ...
          build5050C2(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir, ...
            'kmeans',nCategories);
        save([outDir 'chosenCategories.mat'],'organicCategories','inorganicCategories');
    else
        load([outDir 'chosenCategories.mat'],'organicCategories','inorganicCategories');
        [c2,labels,organicCategories,inorganicCategories,organicC2Files,inorganicC2Files] = ...
          build5050C2(organicImgDir,organicC3Dir,inorganicImgDir,inorganicC3Dir, ...
            'kmeans',nCategories,organicCategories,inorganicCategories);
    end
    save([outDir 'chosenC2Files.mat'],'organicC2Files','inorganicC2Files');
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
    clear c2 labels aucs dprimes models classVals;
else
    load([outDir 'splits-5050.mat'],'cvsplit');
    load([outDir 'chosenC2Files.mat'],'organicC2Files','inorganicC2Files');
end
fprintf('kmeans 50/50 evaluated\n\n');

c2Files = {organicC2Files{:} inorganicC2Files{:}};
c3Files = regexprep(c2Files,'c2','c3');

% cache organic C3
organicC3Files = regexprep(c3Files,'kmeans','organicOldSchoolv2'); % change after commit
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
inorganicC3Files = regexprep(c3Files,'kmeans','inorganicOldSchoolv2'); % change after commit
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

save([outDir 'chosenC3Categories.mat'],'organicC3Files','inorganicC3Files');

% organic
if ~exist([outDir 'organic-evaluation.mat'],'file')
    [c3,labels] = build5050C3(organicC3Files);
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', 'organicC3Files', '-v7.3');
    clear c3 aucs dprimes models classVals;
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
      'models', 'classVals', 'organicC3Files', '-v7.3');
    clear c3 aucs dprimes models classVals;
end
fprintf('organic-super evaluated\n');

% inorganic
if ~exist([outDir 'inorganic-evaluation.mat'],'file')
    [c3,labels] = build5050C3(inorganicC3Files);
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', 'inorganicC3Files', '-v7.3');
    clear c3 aucs dprimes models classVals;
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
      'models', 'classVals', 'inorganicC3Files', '-v7.3');
    clear c3 aucs dprimes models classVals;
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
    clear c3 aucs dprimes models classVals;
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
    clear c3 aucs dprimes models classVals;
end
fprintf('combined-super evaluated\n');

% semantic analysis

nPairs = 1000;
N = 25;
blockSize = [6 6];
maxSize = 240;
featureDir = ensureDir([outDir 'featureCorrelations/']);
simFile = [home 'src/external/similarity.pl'];
load([outDir 'kmeans-5050-evaluation.mat'],'c2');
targetWnids = listImageNetCategories(c2Files);

% random C2 vs category listing check
randomI = randi(length(c2Files));
randC2 = load(c2Files{randomI},'c2');
assert(isequal(randC2.c2,c2(:,((randomI*150-149):(randomI*150)))),'semantic analysis: list is out of order');

% organic analysis
organicDir = ensureDir([featureDir 'organic/']);
[c3,labels,imgNames] = build5050C3(organicC3Files);
imgNames = regexprep(imgNames,'joshrule','josh');
load([organicC3Dir 'splits.mat'],'trainFiles');
organicWnids = listImageNetCategories(trainFiles);
outFile = [organicDir 'pairwiseCorrelations.mat'];
if ~exist(outFile,'file')
    pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
end
fprintf('finished the organic pairwise correlations\n');
categoricalFeatureCorrelations(labels,imgNames,c2,c3,organicWnids, ...
  targetWnids,N,organicDir,simFile,blockSize,maxSize);
fprintf('finished the organic categorical correlations\n');

% inorganic analysis
inorganicDir = ensureDir([featureDir 'inorganic/']);
[c3,labels,imgNames] = build5050C3(inorganicC3Files);
imgNames = regexprep(imgNames,'joshrule','josh');
load([inorganicC3Dir 'splits.mat'],'trainFiles');
inorganicWnids = listImageNetCategories(trainFiles);
outFile = [inorganicDir 'pairwiseCorrelations.mat'];
if ~exist(outFile,'file')
    pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
end
fprintf('finished the inorganic pairwise correlations\n');
categoricalFeatureCorrelations(labels,imgNames,c2,c3,inorganicWnids, ...
  targetWnids,N,inorganicDir,simFile,blockSize,maxSize);
fprintf('finished the inorganic categorical correlations\n');
