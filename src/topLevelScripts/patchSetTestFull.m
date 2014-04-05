home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

outDir   = ensureDir([home 'evaluation/oldFullHNM/']);
patchDir = [home 'patchSets/'];
imgDir   = [home 'imageSets/imageNet/'];
organicImgDir   = [imgDir   'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
nCategories = 50;

if ~exist([outDir 'chosenCategories.mat'],'file')
    organicC3File = [patchDir 'organicC3OldFullHNM.isolated.mat'];
    inorganicC3File = [patchDir 'inorganicC3OldFullHNM.isolated.mat'];
    [organicC2Files,inorganicC2Files] = choose5050Categories(organicImgDir, ...
      organicC3File,inorganicImgDir, inorganicC3File,'kmeans',nCategories);
    save([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
else
    load([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
end
[c2,labels] = responsesFromCaches({organicC2Files{:} inorganicC2Files{:}},'c2');

if ~exist([outDir 'splits.mat'],'file')
    nTrainingExamples = [16 32 64 128 256];
    nRuns = 20;
    cvsplit = cv(labels,nTrainingExamples,nRuns);
    save([outDir 'splits.mat'],'nTrainingExamples','nRuns','cvsplit');
    fprintf('50/50 splits generated\n');
else
    load([outDir 'splits.mat'])
    fprintf('50/50 splits loaded\n');
end

if ~exist([outDir 'kmeans-evaluation.mat'],'file')
    [aucs,dprimes,models,classVals] = evaluatePerformance(c2,labels,cvsplit, ...
      method,options,size(c2,1),[]);
    save([outDir 'kmeans-evaluation.mat'],'labels','c2','aucs', ...
      'dprimes', 'models', 'classVals', '-v7.3');
    clear c2 labels aucs dprimes models classVals;
end
fprintf('kmeans 50/50 evaluated\n');

c2Files = {organicC2Files{:} inorganicC2Files{:}};

% cache C3 activations
type1 = {'organic','inorganic'};
type2 = {'isolated'}; % ,'shared'}; % repair!
for i = 1:length(type1)
    for j = 1:length(type2)
        c3Files{i,j} = regexprep(c2Files,'kmeans.c2', ...
	                         [type1{i} 'OldFullHNM.' type2{j} '.c3']);
        patchFile{i,j} = [patchDir type1{i} 'C3OldFullHNM.' type2{j} '.mat'];
        load(patchFile{i,j},'models');
        for ii = 1:length(c3Files{i,j})
            if ~exist(c3Files{i,j}{ii},'file')
                cacheC3(c3Files{i,j}{ii},c2Files{ii},patchFile{i,j}, ...
		  patchFile{i,j},models);
            end
        end
	clear models;
	fprintf('    %s - %s cached',type1{i},type2{j});
    end
end
fprintf('C3 Activations Cached')
