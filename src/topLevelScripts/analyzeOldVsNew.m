home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

outDir = [home 'evaluation/v3v5Comparison/'];
oldDir = [home 'evaluation/5050v3/'];
newDir = [home 'evaluation/5050v5/'];
oldOrganicPatchDir   = [home 'patchSets/organicC3v2/'];
oldInorganicPatchDir = [home 'patchSets/inorganicC3v2/'];
newOrganicPatchDir   = [home 'patchSets/organicC3v3/'];
newInorganicPatchDir = [home 'patchSets/inorganicC3v3/'];
oldOrganicPatchSet = '.organicOldSchoolv2.c3.mat';
oldInorganicPatchSet = '.inorganicOldSchoolv2.c3.mat';
newOrganicPatchSet = '.organicOldSchoolv3.c3.mat';
newInorganicPatchSet = '.inorganicOldSchoolv3.c3.mat';
imgDir = [home 'imageSets/imageNet/'];
organicImgDir   = [imgDir   'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
nCategories = 50;

fprintf('\n');
fprintf('Initialized\n\n');

rng(0,'twister');
fprintf('Pseudorandom Number Generator Reset\nrng(0,''twister'')\n\n');

% read the category splits
load([home 'evaluation/v3v5Comparison/v3v5Comparison.mat']);

% read the old results
oldC2 = load([oldDir 'kmeans-5050-evaluation.mat'],'c2','labels');

% read the new results
newC2 = load([newDir 'kmeans-5050-evaluation.mat'],'c2','labels');

% list out the shared, old, and new C3 categories
oldOrganicCategories = union(o3Too2,o3Toou);
oldInorganicCategories = union(i3Toi2,i3Toiu);
sharedOrganicCategories = setdiff(o3o,oldOrganicCategories);
sharedInorganicCategories = setdiff(i3o,oldInorganicCategories);
newOrganicCategories = setdiff(o3n,sharedOrganicCategories);
newInorganicCategories = setdiff(i3n,sharedInorganicCategories);
fprintf('For old organic C3 categories, %d stayed, with %d old and %d new categories\n',length(sharedOrganicCategories),length(oldOrganicCategories),length(newOrganicCategories));
fprintf('For old inorganic C3 categories, %d stayed, with %d old and %d new categories\n',length(sharedInorganicCategories),length(oldInorganicCategories),length(newInorganicCategories));

% figure out where the shared, old, and new categories are in the activations
load([oldOrganicPatchDir 'splits.mat'],'trainFiles');
oldSharedOrganicModelIndices = ismember(listImageNetCategories(trainFiles),sharedOrganicCategories);
oldOrganicModelIndices = ismember(listImageNetCategories(trainFiles),oldOrganicCategories);
clear trainFiles
load([oldInorganicPatchDir 'splits.mat'],'trainFiles');
oldSharedInorganicModelIndices = ismember(listImageNetCategories(trainFiles),sharedInorganicCategories);
oldInorganicModelIndices = ismember(listImageNetCategories(trainFiles),oldInorganicCategories);
clear trainFiles
load([newOrganicPatchDir 'splits.mat'],'trainFiles');
newSharedOrganicModelIndices = ismember(listImageNetCategories(trainFiles),sharedOrganicCategories);
newOrganicModelIndices = ismember(listImageNetCategories(trainFiles),newOrganicCategories);
clear trainFiles
load([newInorganicPatchDir 'splits.mat'],'trainFiles');
newSharedInorganicModelIndices = ismember(listImageNetCategories(trainFiles),sharedInorganicCategories);
newInorganicModelIndices = ismember(listImageNetCategories(trainFiles),newInorganicCategories);
clear trainFiles

% build filenames
oldC3Cats = load([oldDir 'chosenC3Categories.mat']);
newC3Cats = load([newDir 'chosenC3Categories.mat']);
oldOrganicC3Files   = oldC3Cats.organicC3Files;
oldInorganicC3Files = oldC3Cats.inorganicC3Files;
newOrganicC3Files   = newC3Cats.organicC3Files;
newInorganicC3Files = newC3Cats.inorganicC3Files;
fprintf('all files loaded\n\n');

% build cvsplits
if ~exist([outDir 'splits.mat'],'file')
    nTrainingExamples = [16 32 64 128 256];
    nRuns = 20;
    cvsplit = cv(oldC2.labels,nTrainingExamples,nRuns);
    save([outDir 'splits.mat'],'nTrainingExamples','nRuns','cvsplit');
    fprintf('50/50 splits generated\n\n');
else
    load([outDir 'splits.mat'])
    fprintf('splits loaded\n\n');
end

if ~exist([outDir 'old-complete-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldOrganicC3Files);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-complete-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-complete-organic evaluated\n');

if ~exist([outDir 'old-complete-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldInorganicC3Files);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-complete-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-complete-inorganic evaluated\n');

if ~exist([outDir 'old-shared-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldOrganicC3Files);
    c3 = c3(oldSharedOrganicModelIndices,:);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-shared-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-shared-organic evaluated\n');

if ~exist([outDir 'old-shared-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldInorganicC3Files);
    c3 = c3(oldSharedInorganicModelIndices,:);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-shared-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-shared-inorganic evaluated\n');

if ~exist([outDir 'old-unique-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldOrganicC3Files);
    c3 = c3(oldOrganicModelIndices,:);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-unique-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-unique-organic evaluated\n');

if ~exist([outDir 'old-unique-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(oldInorganicC3Files);
    c3 = c3(oldInorganicModelIndices,:);
    c2 = oldC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'old-unique-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('old-unique-inorganic evaluated\n');

if ~exist([outDir 'new-complete-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newOrganicC3Files);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-complete-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-complete-organic evaluated\n');

if ~exist([outDir 'new-complete-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newInorganicC3Files);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-complete-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-complete-inorganic evaluated\n');

if ~exist([outDir 'new-shared-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newOrganicC3Files);
    c3 = c3(newSharedOrganicModelIndices,:);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-shared-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-shared-organic evaluated\n');

if ~exist([outDir 'new-shared-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newInorganicC3Files);
    c3 = c3(newSharedInorganicModelIndices,:);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-shared-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-shared-inorganic evaluated\n');

if ~exist([outDir 'new-unique-organic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newOrganicC3Files);
    c3 = c3(newOrganicModelIndices,:);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-unique-organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-unique-organic evaluated\n');

if ~exist([outDir 'new-unique-inorganic-evaluation.mat'],'file') %%%%
    [c3,labels] = build5050C3(newInorganicC3Files);
    c3 = c3(newInorganicModelIndices,:);
    c2 = newC2.c2;
    c3 = [c3; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save([outDir 'new-unique-inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
    clear c3 c2 aucs dprimes models classVals;
end
fprintf('new-unique-inorganic evaluated\n');

for ii = 1:5
    order = randperm(100);
    if ~exist([outDir 'new-organic-random-' num2str(ii) '-evaluation.mat'],'file') %%%%
        [c3,labels] = build5050C3(newOrganicC3Files(order));
        c2 = newC2.c2;
        c3 = [c3; c2];
        [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
          method,options,size(c3,1),[]);
        save([outDir 'new-organic-random-' num2str(ii) '-evaluation.mat'],'labels','c3','aucs','dprimes', ...
          'models', 'classVals', 'order', 'newOrganicC3Files', '-v7.3');
        clear c3 c2 aucs dprimes models classVals;
    end
    fprintf('new-organic-random-%d evaluated\n',ii);
    
    if ~exist([outDir 'new-inorganic-random-' num2str(ii) '-evaluation.mat'],'file') %%%%
        [c3,labels] = build5050C3(newInorganicC3Files(order));
        c2 = newC2.c2;
        c3 = [c3; c2];
        [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
          method,options,size(c3,1),[]);
        save([outDir 'new-inorganic-random-' num2str(ii) '-evaluation.mat'],'labels','c3','aucs','dprimes', ...
          'models', 'classVals', 'order', 'newInorganicC3Files', '-v7.3');
        clear c3 c2 aucs dprimes models classVals;
    end
    fprintf('new-inorganic-random-%d evaluated\n',ii);
    clear order;
end
