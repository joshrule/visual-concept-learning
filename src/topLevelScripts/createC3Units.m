function createC3Units(outDir,params)
% createC3Units(outDir,params)
%
% Given the parameters, load cached C2 responses and generate C3 classifiers.
%
% outDir: string, directory to which to write the simulation data
% params: struct, the parameters governing the simulation itself with the
% following structure:
%   home: top-level directory holding code, data, etc.
%   c2Dir: string, location of training class c2 caches
%   nModels: scalar, number of training categories (i.e. vocabulary concepts)
%   patchSet: string, what patch set should the classes have been cached with?
%   minPerClass: scalar, minimum number of examples allowed in each training and
%     testing class
%   trainOptions: struct with the following fields, for training the c3 units
%       svmTrainFlags: options for training C3 classifiers with hard mining
%       svmTestFlags: options for testing C3 classifiers with hard mining
%       alpha: scalar, governs the growth of the hard negative mining
%       startPerIter: scalar, number of images in the first mining iteration
%       threshold: scalar, probability above which a negative is "hard"
%   testOptions: string or double, options for the test classifier
%   nTrainingExamples: double array, total numbers of training examples to use
%     in evaluation (e.g. [16 32 64 128])
%   method: string, 'gb' or 'svm', the classifier to use

    p = params;
    if ~exist([outDir 'setup.mat'],'file')
        rngState = rng;
        ensureDir(outDir);
        [files,c2,labels] = prepareC3(p.c2Dir,p.patchSet,p.minPerClass,p.nModels);
        save([outDir 'setup.mat'],'params','c2','labels','files','rngState','-v7.3');
    else
        load([outDir 'setup.mat']);
    end

    if ~exist([outDir 'models.mat'],'file')
        rngState = rng;
        models = trainC3(c2,labels,p.method,p.trainOptions,p.repRatio,p.mining);
        save([outDir 'models.mat'],'rngState','models','-v7.3');
        fprintf('%s created\n',[outDir 'models.mat']);
    else
        fprintf('%s found\n',[outDir 'models.mat']);
    end
end

function [nFiles,nC2,nLabels] = prepareC3(cacheDir,patchSet,minImgs,N)
% [nFiles,nC2,nLabels] = prepareC3(cacheDir,patchSet,minImgs,N)
%
% the actual work of preparing the splits
%
% cacheDir: string, the directory holding relevant cacheFiles
% patchSet: string, only use caches created with this patch set
% minImgs: scalar, the minimum number of images required to be useful
% N: scalar, number of cacheFiles/categories to use
%
% nFiles: cell array of strings, the cache files used
% nC2: nFeatures x nImgs array, C2 matrix made by concatenating the cache files
% nLabels: N x nImgs array, class labelings for each image
    caches = dir([cacheDir '*' patchSet '.c2.mat']);
    cacheFiles = strcat(cacheDir,{caches.name}');
    shuffledClasses = cacheFiles(randperm(length(cacheFiles)));
    goodClasses = checkForMinExamples(shuffledClasses,minImgs);
    nFiles = shuffledClasses(goodClasses(1:(min(length(goodClasses),N))));

    nClasses = length(nFiles);
    nImgs = zeros(nClasses,1);
    c2s = cell(nClasses,1);
    for iClass = 1:nClasses
        load(nFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
        c2s{iClass} = c2;
        clear c2;
    end
    nLabels = zeros(nClasses,sum(nImgs));
    for iClass = 1:nClasses
        start = sum(nImgs(1:iClass-1))+1;
        stop = start+nImgs(iClass)-1;
        nLabels(iClass,start:stop) = 1;
    end
    nC2 = [c2s{:}];
end

function [goodClasses,nImgs] = checkForMinExamples(cacheFiles,minPerClass)
% [goodClasses,nImgs] = checkForMinExamples(cacheFiles,minPerClass)
%
% check a cacheFile to ensure it has enough examples to be useful
%
% cacheFiles: cell array of strings, a list of c2 cache files
% minPerClass: scalar, the minimum number of images required to be useful
%
% goodClasses: cell array of strings, the cache files with minPerClass images
% nImgs: the actual number of images in each cache file listed in 'goodClasses'
    nClasses = length(cacheFiles);
    nImgs = zeros(nClasses,1);
    for iClass = 1:nClasses
        load(cacheFiles{iClass},'c2');
        nImgs(iClass) = size(c2,2);
    end
    goodClasses = find(nImgs >= minPerClass);
    nImgs = nImgs(goodClasses);
end
