home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

outDir   = ensureDir([home 'evaluation/oldSuperHNM/']);
patchDir = [home 'patchSets/'];
imgDir   = [home 'imageSets/imageNet/'];
organicImgDir   = [imgDir   'organicC2Cache/'];
inorganicImgDir = [imgDir 'inorganicC2Cache/'];

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
nCategories = 50;

if ~exist([outDir 'chosenCategories.mat'],'file')
    organicC3File = [patchDir 'organicC3OldHalfHNM.isolated.mat'];
    inorganicC3File = [patchDir 'inorganicC3OldHalfHNM.isolated.mat'];
    [organicC2Files,inorganicC2Files] = choose5050Categories(organicImgDir, ...
      organicC3File,inorganicImgDir, inorganicC3File,'kmeans',nCategories);
    save([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
    fprintf('Categories chosen\n');
else
    load([outDir 'chosenCategories.mat'],'organicC2Files','inorganicC2Files');
    fprintf('Categories loaded\n');
end

organicC2Files = [regexprep(organicC2Files,'Half','')',...
                  organicC2Files',...
                  regexprep(organicC2Files,'Half','Full')',...
                  regexprep(organicC2Files,'Half','Double')'];
inorganicC2Files = [regexprep(inorganicC2Files,'Half','')',...
                    inorganicC2Files',...
                    regexprep(inorganicC2Files,'Half','Full')',...
                    regexprep(inorganicC2Files,'Half','Double')'];
[c2,labels] = responsesFromCaches([organicC2Files; inorganicC2Files],'c2');
fprintf('C2 loaded\n');

if ~exist([outDir 'splits.mat'],'file')
    nTrainingExamples = [16 32 64 128 256];
    nRuns = 20;
    cvsplit = cv(labels,nTrainingExamples,nRuns);
    save([outDir 'splits.mat'],'nTrainingExamples','nRuns','cvsplit');
    fprintf('splits generated\n');
else
    load([outDir 'splits.mat'])
    fprintf('splits loaded\n');
end

if ~exist([outDir 'kmeans-evaluation.mat'],'file')
    [aucs,dprimes,models,classVals] = evaluatePerformance(c2,labels,cvsplit, ...
      method,options,size(c2,1),[]);
    save([outDir 'kmeans-evaluation.mat'],'labels','c2','aucs', ...
      'dprimes', 'models', 'classVals', '-v7.3');
    clear c2 labels aucs dprimes models classVals;
end
fprintf('kmeans 50/50 evaluated\n');

c2Files = [organicC2Files; inorganicC2Files];

% cache C3 activations
type1 = {'inorganic'}; %,'organic'}; % repair!
type2 = {'isolated'}; % ,'shared'}; % repair!
for i = 1:length(type1)
    for j = 1:length(type2)
        c3Files{i,j} = regexprep(c2Files,'kmeans.c2', ...
	                         [type1{i} 'OldHalfHNM.' type2{j} '.c3']);
%         patchFile{i,j} = [patchDir type1{i} 'C3OldHalfHNM.' type2{j} '.mat'];
%         load(patchFile{i,j},'models');
%         for ii = 1:length(c3Files{i,j})
%             if ~exist(c3Files{i,j}{ii},'file')
%                 cacheC3(c3Files{i,j}{ii},c2Files{ii},patchFile{i,j}, ...
% 		  patchFile{i,j},models);
%             end
%         end
% 	clear models;
% 	fprintf('    %s - %s cached',type1{i},type2{j});
    end
end
fprintf('C3 Activations Cached')

for j = 1:length(type2)
    for i = 1:length(type1)
        % single patch type
        outFile = [outDir type1{i} '-' type2{j} '-evaluation.mat'];
        if ~exist(outFile,'file')
            [c3,labels] = responsesFromCaches(c3Files{i,j},'c3');
            [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels, ...
	      cvsplit,method,options,size(c3,1),[]);
            save(outFile,'labels','c3','aucs','dprimes', ...
              'models', 'classVals', '-v7.3');
            clear c3 aucs dprimes models classVals;
        end
        fprintf('%s - %s evaluated\n',type1{i},type2{j});

	% combined patch types
        outFile = [outDir type1{i} '-' type2{j} '-plus-kmeans-evaluation.mat'];
        if ~exist(outFile,'file')
            [c3,labels] = responsesFromCaches(c3Files{i,j},'c3');
            load([outDir 'kmeans-evaluation.mat'],'c2');
            c3 = [c3; c2];
            [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels, ...
	      cvsplit,method,options,size(c3,1),[]);
            save(outFile,'labels','c3','aucs','dprimes', ...
              'models', 'classVals', '-v7.3');
            clear c3 aucs dprimes models classVals;
        end
        fprintf('%s - %s + k-means evaluated\n',type1{i},type2{j});
    end
    % all isolated/shared c3
    outFile = [outDir type2{j} '-all-C3-evaluation.mat'];
    if ~exist(outFile,'file')
    [c3a,~]      = responsesFromCaches(c3Files{1,j},'c3');
    [c3b,labels] = responsesFromCaches(c3Files{2,j},'c3');
    c3 = [c3a; c3b];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save(outFile,'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
        clear c3 aucs dprimes models classVals;
    end
    fprintf('%s all C3 evaluated\n',type2{j});

    % all isolated/shared c3 + k-means
    outFile = [outDir type2{j} '-all-C3--plus-kmeans-evaluation.mat'];
    if ~exist(outFile,'file')
    [c3a,~]      = responsesFromCaches(c3Files{1,j},'c3');
    [c3b,labels] = responsesFromCaches(c3Files{2,j},'c3');
    load([outDir 'kmeans-evaluation.mat'],'c2');
    c3 = [c3a; c3b; c2];
    [aucs,dprimes,models,classVals] = evaluatePerformance(c3,labels,cvsplit, ...
      method,options,size(c3,1),[]);
    save(outFile,'labels','c3','aucs','dprimes', ...
      'models', 'classVals', '-v7.3');
        clear c3 aucs dprimes models classVals;
    end
    fprintf('%s all C3 + kmeans evaluated\n',type2{j});
end

% semantic analysis
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
% [c3,labels,imgNames] = buildC3(organicC3Files);
% imgNames = regexprep(imgNames,'joshrule','josh/data');
% load([organicC3Dir 'splits.mat'],'testFiles');
% organicWnids = listImageNetCategories(testFiles);
% outFile = [organicDir 'pairwiseCorrelations.mat'];
% pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
% categoricalFeatureCorrelations(labels,imgNames,c2,c3,organicWnids, ...
%   targetWnids,N,outDir,simFile,blockSize,maxSize);
% 
% inorganicDir = ensureDir([featureDir 'inorganic/']);
% [c3,labels,imgNames] = buildC3(inorganicC3Files);
% imgNames = regexprep(imgNames,'joshrule','josh/data');
% load([inorganicC3Dir 'splits.mat'],'testFiles');
% inorganicWnids = listImageNetCategories(testFiles);
% outFile = [inorganicDir 'pairwiseCorrelations.mat'];
% pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
% categoricalFeatureCorrelations(labels,imgNames,c2,c3,inorganicWnids, ...
%   targetWnids,N,outDir,simFile,blockSize,maxSize);
