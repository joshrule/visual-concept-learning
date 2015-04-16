function semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
% semanticAnalysis(p,organicC3Files,inorganicC3Files)
    c2 = buildC2(c2Files);
    semanticAnalysisHelper(p,c2,organicC3Files,'organic');
    semanticAnalysisHelper(p,c2,inorganicC3Files,'inorganic');
end

function semanticAnalysisHelper(p,c2,c3Files,type)
    outDir = ensureDir([p.featureDir type '/']);
    [c3,labels,imgNames] = buildC3(c3Files);

    pairFile = [outDir 'pairwiseCorrelations.mat'];
    if ~exist(pairFile,'file')
        pairwiseFeatureCorrelations(imgNames,c2,c3,p.nPairs, ...
            p.caching.maxSize,p.simFile,pairFile);
    end
    fprintf(['finished the ' type ' pairwise correlations\n']);

    semDistFile = [outDir 'semanticDistances.mat'];
    if ~exist(semDistFile,'file')
        semanticDistances = c2Vc2SemanticDistances(p,c3Files);
        save(semDistFile,'semanticDistances');
    else
        load(semDistFile,'semanticDistances');
    end
    fprintf(['finished all ' type ' pairwise distances\n']);

    semDistFile2 = [outDir 'c3semanticDistances.mat'];
    if ~exist(semDistFile2,'file')
        semanticDistances = c2Vc3SemanticDistances(p,c3Files,type);
        save(semDistFile2,'semanticDistances');
    else
        load(semDistFile2,'semanticDistances');
    end
    fprintf(['finished all ' type ' C3 distances\n']);

    diffFile = [outDir 'dPrimeDiffs.mat'];
    if ~exist(diffFile,'file')
        data = printEvaluations(p.outDir,'dprimes',0,0);
        kmeans = load([p.outDir 'kmeans-evaluation.mat'],'dprimes');
        kmeans = kmeans.dprimes;
        for i = 1:size(data,1)
            diff(i,:,:,:) = squeeze(data(i,:,:,:)) - kmeans;
        end
        save(diffFile,'data','kmeans','diff');
    else
        load(diffFile,'data','kmeans','diff');
    end
    fprintf(['finished computing ' type ' performance relative to C2\n']);

    dprimeFile = [outDir 'featureDPrimes.mat'];
    if ~exist(dprimeFile,'file')
        [aucs,dprimes] = vocabularyDPrimes(c3,labels);
        save(dprimeFile,'aucs','dprimes');
    else
        load(dprimeFile,'aucs','dprimes');
    end
    fprintf(['finished computing ' type ' C3-feature d-primes\n']);
end

function [aucs,dprimes] = vocabularyDPrimes(c3,labels)
    for iC3 = 1:size(c3,1)
        if mod(iC3,10) == 0, fprintf('%d...',iC3); end;
        parfor iClass = 1:size(labels,1)
            aucs(iC3,iClass) = auc(c3(iC3,:),labels(iClass,:));
            dprimes(iC3,iClass) = myDPrime(c3(iC3,(labels(iClass,:)==1)),c3(iC3,(labels(iClass,:)==0)));
        end
    end
end

function d  = myDPrime(x,y)
    d = (mean(x) - mean(y))/sqrt(1/2*var(x)+var(y));
end

function semanticDistance = c2Vc2SemanticDistances(p,c3Files)
    rows = listImageNetCategories(c3Files);
    cols = rows;
    for iR = 1:length(rows)
        for iC = 1:length(cols)
            semanticDistance(iR,iC) = str2num(perl(p.simFile,rows{iR},cols{iC}));
        end
    end
end

function semanticDistance = c2Vc3SemanticDistances(p,c3Files,type)
    cols = listImageNetCategories(c3Files);
    a = load([p.outDir 'c3Vocabulary.mat'],[type 'C3Vocab']);
    rows = getfield(a,[type 'C3Vocab']);
    for iC = 1:length(cols)
        if mod(iC,10) == 0, fprintf('%d...',iC); end;
        parfor iR = 1:length(rows)
            semanticDistance(iR,iC) = str2num(perl(p.simFile,rows{iR},cols{iC}));
        end
    end
    fprintf('\n');
end
