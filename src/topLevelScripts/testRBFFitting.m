function testRBFFitting(pHandle)
    p = pHandle();
    fprintf('Params Initialized\n\n');

    addpath(genpath(p.srcPath));
    fprintf('Source Loaded\n\n');

    rng(p.seed,'twister');
    fprintf('Pseudorandom Number Generator Reset\n\n');

    % cache C2 activations
    cacheC2Wrapper(p);

    % write out C3 SVM stuff
    if ~exist([p.outDir 'fittingData.mat'],'file')
        [trainL,trainD,testL,testD] = setupC3(p.organicC3Dir,p.organicC3Params,'o');
        save([p.outDir 'fittingData.mat'],'trainL','trainD','testL','testD','-v7.3');
    else
        load([p.outDir 'fittingData.mat'],'trainL','trainD','testL','testD');
    end

    if ~exist([p.outDir 'fittingResultFixed.mat'],'file')
        fixedAccs = zeros(length(trainL),1);
        fixedTs = zeros(length(trainL),1);
        parfor iClass = 1:length(trainL)
            a = tic;
            fixedAccs(iClass) = testFixedSVM(trainL{iClass},trainD{iClass},testL{iClass},testD{iClass});
            fixedTs(iClass) = toc(a);
            fprintf('fixed time: %.2f\n\n',fixedTs(iClass));
        end
        save([p.outDir 'fittingResultFixed.mat'], 'fixedAccs','fixedTs','-v7.3');
    end

    if ~exist([p.outDir 'fittingResultLinear.mat'],'file')
        linearAccs = zeros(length(trainL),1);
        linearTs = zeros(length(trainL),1);
        parfor iClass = 1:length(trainL)
            a = tic;
            linearAccs(iClass) = testFixedSVM2(trainL{iClass},trainD{iClass},testL{iClass},testD{iClass});
            linearTs(iClass) = toc(a);
            fprintf('fixed time: %.2f\n\n',linearTs(iClass));
        end
        save([p.outDir 'fittingResultLinear.mat'],'linearTs','linearAccs','-v7.3');
    end

    if ~exist([p.outDir 'fittingResultSearch.mat'],'file')
        searchAccs = zeros(length(trainL),1);
        searchTs = zeros(length(trainL),1);
        c = zeros(length(trainL),1);
        g = zeros(length(trainL),1);
        parfor iClass = 1:length(trainL)
            a = tic;
            [searchAccs(iClass),c(iClass),g(iClass)] = ...
              testSearchSVM(trainL{iClass},trainD{iClass},testL{iClass},testD{iClass});
            searchTs(iClass) = toc(a);
            fprintf('search time: %.2f\n\n',searchTs(iClass));
        end
        save([p.outDir 'fittingResultSearch.mat'],'searchAccs','c','g','searchTs','-v7.3');
    end
    fixMyGoof(trainL,trainD,testL,testD);
end

function [trainL,trainD,testL,testD] = setupC3(outDir,params,type)
    p = params;
    if ~exist([outDir 'setup.mat'],'file')
        rngState = rng;
        ensureDir(outDir);
        [files,c2,labels] = prepareC3(p.c2Dir,p.patchSet,p.minPerClass,p.nModels);
        [nClasses, nImgs] = size(labels);
        for iClass = 1:min(20,nClasses)
            fprintf('%d/%d\n',iClass,nClasses);
            training = equalRep(labels(iClass,:),100,p.repRatio);
            trainD{iClass} = c2(:,training)';
            trainL{iClass} = labels(iClass,training)';
            preTestX = c2(:,~training)';
            preTestY = labels(iClass,~training)';
            testing = equalRep(preTestY,inf,p.repRatio);
            testD{iClass} = preTestX(testing,:);
            testL{iClass} = preTestY(testing);
        end
        save([outDir 'setup.mat'],'params','c2','labels','files','rngState','-v7.3');
    else
        load([outDir 'setup.mat']);
    end
end

function acc = testFixedSVM(trainL,trainD,testL,testD)
    cmd = ['-b 1 -s 0 -t 2 -q'];
    [trainD,minVals,maxVals] = libsvmScaleData(trainD,0,1);
    testD = libsvmScaleData(testD,0,1,minVals,maxVals);
    model = svmtrain(trainL, trainD, cmd);
    [pred,acc] = svmpredict(testL,testD,model,'-b 1');
    acc = acc(1);
    fprintf('accuracy is %.2f\n',acc);
end

function acc = testFixedSVM2(trainL,trainD,testL,testD)
    cmd = ['-b 1 -s 0 -t 0 -c 0.1 -q'];
    [trainD,minVals,maxVals] = libsvmScaleData(trainD,0,1);
    testD = libsvmScaleData(testD,0,1,minVals,maxVals);
    model = svmtrain(trainL, trainD, cmd);
    [pred,acc] = svmpredict(testL,testD,model,'-b 1');
    acc = acc(1);
    fprintf('accuracy is %.2f\n',acc);
end

function [acc,bestc,bestg] = testSearchSVM(trainL,trainD,testL,testD)
    bestcv = 0;
    [trainD,minVals,maxVals] = libsvmScaleData(trainD,0,1);
    testD = libsvmScaleData(testD,0,1,minVals,maxVals);
    for log2c = -4:15,
        for log2g = -15:-0,
            cmd = ['-q -s 0 -t 2 -b 1 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = svmtrain(trainL, trainD, cmd);
            if (cv >= bestcv),
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
            end
        end
    end
    fprintf('best c=%g, g=%g, rate=%g\n', bestc, bestg, bestcv);
    cmd = ['-q -b 1 -s 0 -t 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model = svmtrain(trainL, trainD, cmd);
    [pred,acc] = svmpredict(testL,testD,model,'-b 1');
    acc = acc(1);
    fprintf('accuracy is %.2f\n',acc);
end

function fixMyGoof(trainL,trainD,testL,testD)
    load('/home/josh/data/ruleRiesenhuber2013/evaluation/5050vRBF/fittingResultSearch.mat','c','g','searchTs');
    for iClass = 1:length(trainL)
        trD = trainD{iClass}; trL = trainL{iClass};
        teD = testD{iClass}; teL = testL{iClass};
        [trD,minVals,maxVals] = libsvmScaleData(trD,0,1);
        teD = libsvmScaleData(teD,0,1,minVals,maxVals);
        cmd = ['-q -b 1 -s 0 -t 2 -c ', num2str(c(iClass)), ' -g ', num2str(g(iClass))];
        model = svmtrain(trL, trD, cmd);
        [pred,acc] = svmpredict(teL,teD,model,'-b 1');
        searchAccs(iClass) = acc(1);
        fprintf('accuracy is %.2f\n',acc(1));
    end
    save('/home/josh/data/ruleRiesenhuber2013/evaluation/5050vRBF/fittingResultSearchFixed.mat','c','g','searchTs','searchAccs','-v7.3');
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
