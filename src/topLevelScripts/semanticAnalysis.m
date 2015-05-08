function semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
% semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
    contentAnalysis(p,c2Files,organicC3Files,inorganicC3Files);
    classifierAnalysis(p,c2Files);
end

function contentAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
    if exist([p.outDir '/contentAnalysis.mat'],'file') 
        load([p.outDir '/contentAnalysis.mat']);
    end

    c2 = buildC2(c2Files);
    [oc3,olabels,oImageFiles] = buildC3(organicC3Files);
    [ic3,ilabels,iImageFiles] = buildC3(inorganicC3Files);
    c3 = [oc3; ic3];
    assert(isequal(ilabels,olabels), 'Failed label equality in semantic analysis 1\n');
    assert(isequal(iImageFiles,oImageFiles), 'Failed image list equality in semantic analysis 1\n');

    testCategories = listImageNetCategories(c2Files);
    tmp = load([p.outDir 'c3Vocabulary.mat'],'organicC3Vocab','inorganicC3Vocab');
    featureCategories = [tmp.organicC3Vocab; tmp.inorganicC3Vocab];

    if ~(exist('c2Correlations','var') && exist('c2ps','var'))
        [c2Correlations,c2ps] = corr(c2);
        save([p.outDir '/contentAnalysis.mat'],'c2Correlations','c2ps', ...
          'testCategories','featureCategories','-v7.3');
        clear c2ps
    end
    fprintf('Similarity 1 computed\n');

    if ~(exist('c3Correlations','var') && exist('c3ps','var'))
        [c3Correlations,c3ps] = corr(c3);
        save([p.outDir '/contentAnalysis.mat'],'c3Correlations','c3ps','-v7.3','-append');
        clear c3ps
    end
    fprintf('Similarity 2 computed\n');

    if ~exist('testVsTestSemanticSimilarities','var')
        testVsTestSemanticSimilarities = pairwiseScores(p.simFile,testCategories,testCategories);
        save([p.outDir '/contentAnalysis.mat'],'testVsTestSemanticSimilarities','-v7.3','-append');
    end
    fprintf('Similarity 3 computed\n');

    if ~exist('featureVsTestSemanticSimilarities','var')
        featureVsTestSemanticSimilarities = pairwiseScores(p.simFile,featureCategories,testCategories);
        save([p.outDir '/contentAnalysis.mat'],'featureVsTestSemanticSimilarities','-v7.3','-append');
        clear featureVsTestSemanticSimilarities
    end
    fprintf('Similarity 4 computed\n');

    if ~(exist('rs','var') && exist('ps','var'))
        for i = 1:size(c2,2)
            for j = 1:size(c2,2)
                fullSizedTVTSemanticSimilarities(i,j) = ...
                  testVsTestSemanticSimilarities(ceil(i/p.caching.nImgs),ceil(j/p.caching.nImgs));
            end
        end

        idx = find(triu(ones(size(c2Correlations)),1));
        uniqueC2Corrs = triu(c2Correlations,1);
        uC2C = reshape(uniqueC2Corrs(idx),1,[]);

        uniqueC3Corrs = triu(c3Correlations,1);
        uC3C = reshape(uniqueC3Corrs(idx),1,[]);

        uniqueSemanticSims = triu(fullSizedTVTSemanticSimilarities,1);
        uSS = reshape(uniqueSemanticSims(idx),1,[]);

        [rs(1),ps(1)] = corr(c2Correlations(:),c3Correlations(:));
        [rs(2),ps(2)] = corr(c2Correlations(:),fullSizedTVTSemanticSimilarities(:));
        [rs(3),ps(3)] = corr(c3Correlations(:),fullSizedTVTSemanticSimilarities(:));
        save([p.outDir '/contentAnalysis.mat'],'fullSizedTVTSemanticSimilarities','rs','ps','-v7.3','-append');
    end
    fprintf('Correlations computed\n');
    
    % if ~exist('featureVsFeatureSemanticSimilarities','var')
    %     featureVsFeatureSemanticSimilarities = pairwiseScores(p.simFile,featureCategories,featureCategories);
    %     save([p.outDir '/contentAnalysis.mat'],'featureVsFeatureSemanticSimilarities','-v7.3','-append');
    %     clear featureVsFeatureSemanticSimilarities
    % end
    % fprintf('Similarity 5 computed\n');
end

function classifierAnalysis(p,c2Files)
    load([p.home 'evaluation/5050v' p.suffix '/c3Vocabulary.mat'],'organicC3Vocab','inorganicC3Vocab');
    load([p.home 'evaluation/5050v' p.suffix '/chosenCategories.mat'],'organicCategories','inorganicCategories');
    o = organicC3Vocab;
    i = inorganicC3Vocab;
    c = [o;i];
    c2Cats = [organicCategories inorganicCategories]';

    [classifierNFeats,classifierMeanChosen,classifierMeanUnchosen] = ...
      semanticHelper(c,[p.home 'evaluation/5050v' p.suffix '/combined-thresh-0.55-evaluation.mat'],...
      [p.home 'evaluation/5050v' p.suffix '/semantic-analysis/combined-thresh-0.55-analysis.mat']);

    function [nFeats,meanChosen,meanUnchosen] = semanticHelper(wnids,filename,outfile)
        load(filename,'features');
        [nClass,nTrain,nSplit] = size(features);
        [d,f,e] = fileparts(outfile);
        nFeats = nan(size(features));
        meanChosen = nan(size(features));
        meanUnchosen = nan(size(features));
        for iClass = 1:nClass
            for iTrain = 1:nTrain
                for iSplit = 1:nSplit
                    fprintf('%d %d %d/%d %d %d\n',iClass,iTrain,iSplit,nClass,nTrain,nSplit);
                    f2 = [f '-' num2str(iClass) '-' num2str(iTrain) '-' num2str(iSplit)];
                    a = individualAnalysis(p,features{iClass,iTrain,iSplit},c2Cats{iClass},wnids,[d '/' f2 e]);
                    nFeats(iClass,iTrain,iSplit) = a.nFeats;
                    meanChosen(iClass,iTrain,iSplit) = a.meanChosenScores;
                    meanUnchosen(iClass,iTrain,iSplit) = a.meanUnchosenScores;
                end
            end
        end
    end

    if exist([p.outDir '/classifierAnalysis.mat'],'file')
        load([p.outDir '/classifierAnalysis.mat']);
    end

    if ~(exist('classifierNFeats','var') && exist('classifierMeanChosen','var') && ...
         exist('classifierMeanUnchosen','var') && exist('classifierCorrs','var'))
        fprintf('bringing it all together for the classifier analysis\n');
        organicModelData = load([p.organicC3Dir '/setup.mat']);
        inorganicModelData = load([p.inorganicC3Dir '/setup.mat']);
        o2 = listImageNetCategories(organicModelData.files);
        i2 = listImageNetCategories(inorganicModelData.files);
        assert(isequal(i,i2), 'mismatched organic category list!\n');
        assert(isequal(o,o2), 'mismatched inorganic category list!\n');
        classifierCorrs = [];
        for i = 1:length(c2Files)
            testC2 = buildC2(c2Files(i));
            newCorrs = [corr(organicModelData.c2,testC2); corr(inorganicModelData.c2,testC2)];
            classifierCorrs = [classifierCorrs newCorrs];
        end
    else
        fprintf('found the main classifier analysis\n');
    end

    if ~(exist('sampledIncludedCorrelations','var') && exist('sampledExcludedCorrelations','var'))
        fprintf('putting together the lists of sampled correlations\n');
        sampledIncludedCorrelations = cell(nClass,nTrain,nSplit);
        sampledExcludedCorrelations = cell(nClass,nTrain,nSplit);
        sampledIncludedIndices = cell(nClass,nTrain,nSplit);
        sampledExcludedIndices = cell(nClass,nTrain,nSplit);
        for iClass = 1:nClass
            for iTrain = 1:nTrain
                for iSplit = 2:nSplit
                    fprintf('%d %d %d/%d %d %d\n',iClass,iTrain,iSplit,nClass,nTrain,nSplit);
                    sampleRate = 0.01; % 1% sample rate
                    [included,excluded,inIdx,exIdx] = sampleCorrelations(p,sampleRate,iClass,iTrain,iSplit,classifierCorrs(:,(150*iClass-149):(150*iClass)),c);
                    sampledIncludedIndices{iClass,iTrain,iSplit} = inIdx;
                    sampledExcludedIndices{iClass,iTrain,iSplit} = exIdx;
                    sampledIncludedCorrelations{iClass,iTrain,iSplit} = included;
                    sampledExcludedCorrelations{iClass,iTrain,iSplit} = excluded;
                end
            end
        end
    else
        fprintf('loaded the lists of sampled correlations\n');
    end

    save([p.outDir '/classifierAnalysis.mat'],'classifierCorrs', ...
      'classifierNFeats','classifierMeanChosen','classifierMeanUnchosen', ...
      'sampledIncludedCorrelations','sampledExcludedCorrelations','-v7.3');
end

function [in,ex,inIdx,exIdx] = sampleCorrelations(p,rate,iClass,iTrain,iSplit,corrs,classes)
% note that corrs should be contain images from only one classifier (i.e. only 150 columns in my experiments) 
    load ([p.home 'evaluation/5050v' p.suffix '/semantic-analysis/' ...
           'combined-thresh-0.55-analysis-' num2str(iClass) '-'  ...
           num2str(iTrain) '-' num2str(iSplit) '.mat'], 'analysis');
    inCats = find(ismember(classes,analysis.wnids));
    exCats = find(ismember(classes,setdiff(classes,analysis.wnids)));

    allInIdxs = index1DBlocks(corrs,p.caching.nImgs,p.caching.nImgs,inCats);
    allExIdxs = index1DBlocks(corrs,p.caching.nImgs,p.caching.nImgs,exCats);

    inIdx = randperm(length(allInIdxs),floor(length(allInIdxs)*rate));
    exIdx = randperm(length(allExIdxs),floor(length(allExIdxs)*rate));

    in = corrs(inIdx);
    ex = corrs(exIdx);
end

function analysis = individualAnalysis(p,features,cat,wnids,outfile)
    if ~exist(outfile,'file')
        fprintf('computing analysis for %s\n',cat);
        analysis.features = features;
        analysis.name = imageNetName(cat,p.srFile);
        n = length(features);
        analysis.nFeats = n;
        analysis.chosenNames = imageNetNames(wnids(analysis.features),p.srFile);
        unchosenFeatures = setdiff(1:length(wnids),analysis.features);
        analysis.selectedUnchosenFeatures = unchosenFeatures(randi(length(unchosenFeatures),1,n));
        analysis.pairwiseChosenScores = pairwiseScores(p.simFile,{cat},wnids(analysis.features));
        analysis.pairwiseUnchosenScores = pairwiseScores(p.simFile,{cat},wnids(analysis.selectedUnchosenFeatures));
        analysis.meanChosenScores = mean(analysis.pairwiseChosenScores);
        analysis.meanUnchosenScores = mean(analysis.pairwiseUnchosenScores);
        save(outfile,'analysis','-v7.3');
    else
        fprintf('loading analysis for %s\n',cat);
        load(outfile,'analysis');
        analysis.wnids = wnids(analysis.features);
        save(outfile,'analysis');
    end
end

function names = imageNetNames(wnids,srFile)
    names = cell(length(wnids),1);
    parfor i = 1:length(wnids)
        names{i} = imageNetName(wnids{i},srFile);
    end
end

function name = imageNetName(wnid,srFile)
    w = wnidToDefinition(srFile,wnid);
    name = w.words;
end

function semanticScore = pairwiseScores(simFile,wnids1,wnids2)
    semanticScore = nan(length(wnids1),length(wnids2));
    for i1 = 1:length(wnids1)
        if (mod(i1,10)==0), fprintf('%d, ',i1); end;
        parfor i2 = 1:length(wnids2)
            semanticScore(i1,i2) = str2num(perl(simFile,wnids1{i1},wnids2{i2}));
        end
    end
    fprintf('%d\n',length(wnids1));
end
