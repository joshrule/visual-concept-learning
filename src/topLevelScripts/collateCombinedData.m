function [cds,c3Names] = collateCombinedData(dir)
    load([dir 'combined-evaluation.mat'],'models','m','p','labels');
    load([dir 'featureCorrelations/dPrimeDiffs.mat'],'diff');
    o = load([dir 'featureCorrelations/organic/c3semanticDistances.mat'],'semanticDistances');
    i = load([dir 'featureCorrelations/inorganic/c3semanticDistances.mat'],'semanticDistances');
    v = load([dir 'c3Vocabulary.mat'],'organicC3Vocab','inorganicC3Vocab');
    c = load([dir 'chosenCategories.mat'],'organicCategories','inorganicCategories');
    c3Vocab = [v.organicC3Vocab; v.inorganicC3Vocab];
    srFile = '/home2/josh/maxlab/image-sets/image-net/images/structure_released.xml';
    c3Names = listImageNetMeaning(c3Vocab,srFile);
    testCats = [c.organicCategories c.inorganicCategories]';
    sds = [o.semanticDistances; i.semanticDistances];
    nTrainingExamplesOfInterest = 256;
    idxOfInterest = find(p.nTrainingExamples == nTrainingExamplesOfInterest);
    relevantDiffs = squeeze(diff(1,:,idxOfInterest,:)); % hack!
    relevantModels = squeeze(models(:,idxOfInterest,:));
    [nClasses,nSplits] = size(relevantModels);
    cds = struct();
    cds = struct('svs',cell(nClasses,nSplits),...
                 'coefs',cell(nClasses,nSplits),...
                 'ws',cell(nClasses,nSplits),...
                 'gain',cell(nClasses,nSplits),...
                 'scores',cell(nClasses,nSplits),...
                 'wnid',cell(nClasses,nSplits),...
                 'name',cell(nClasses,nSplits),...
                 'idW',cell(nClasses,nSplits),...
                 'idS',cell(nClasses,nSplits));
    parfor iClass = 1:nClasses
        for iSplit = 1:nSplits
            cds(iClass,iSplit).svs = relevantModels{iClass,iSplit}.SVs;
            cds(iClass,iSplit).coefs = relevantModels{iClass,iSplit}.sv_coef;
            ws = relevantModels{iClass,iSplit}.SVs' * relevantModels{iClass,iSplit}.sv_coef;
            if (relevantModels{iClass,iSplit}.Label(1) == -1) ws = -ws; end;
            cds(iClass,iSplit).ws = ws;
            cds(iClass,iSplit).gain = relevantDiffs(iClass,iSplit);
            cds(iClass,iSplit).scores = sds(:,iClass);
            cds(iClass,iSplit).wnid = testCats{iClass};
            cds(iClass,iSplit).name = listImageNetMeaning(testCats(iClass),srFile);
            [~,idW] = sort(cds(iClass,iSplit).ws,'descend');
            cds(iClass,iSplit).idW = idW;
            [~,idS] = sort(cds(iClass,iSplit).scores,'descend');
            cds(iClass,iSplit).idS = idS;
        end
    end
end

function meanings = listImageNetMeaning(wnids,srFile)
    for i = 1:length(wnids)
        w = wnidToDefinition(srFile,wnids{i});
        meanings{i} = sprintf('%s :: %s',w.words,w.gloss);
    end
end
