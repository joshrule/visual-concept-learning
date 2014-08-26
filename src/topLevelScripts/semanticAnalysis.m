function semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
% semanticAnalysis(p,organicC3Files,inorganicC3Files)
    c2 = buildC2(c2Files);
    semanticAnalysisHelper(p,c2,organicC3Files,'organic')
    semanticAnalysisHelper(p,c2,inorganicC3Files,'inorganic')
end

function semanticAnalysisHelper(p,c2,c3Files,type)
    outDir = ensureDir([p.featureDir type '/']);
    [c3,labels,imgNames] = buildC3(c3Files);
    outFile = [outDir 'pairwiseCorrelations.mat'];
    if ~exist(outFile,'file')
        pairwiseFeatureCorrelations(imgNames,c2,c3,p.nPairs, ...
            p.caching.maxSize,p.simFile,outFile);
    end
    fprintf('finished the ' type ' pairwise correlations\n');
end
