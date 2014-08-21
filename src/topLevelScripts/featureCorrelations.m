nPairs = 1000;
N = 25;
blockSize = [6 6];
maxSize = 240;
targetWnids = listImageNetCategories(c3Files);
featureDir = ensureDir([outDir 'featureCorrelations/']);
simFile = [home 'external/similarity.pl'];
load([outDir 'kmeans-5050-evaluation.mat'],'c2');

organicDir = ensureDir([featureDir 'organic/']);
if ~exist(organicDir,'dir')
  [c3,labels,imageNames] = build5050C3(organicC3Files);
  load([organicC3Dir 'splits.mat'],'testFiles');
  organicWnids = listImageNetCategories(testFiles);
  outFile = [organicDir 'pairwiseCorrelations.mat'];
  pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
  categoricalFeatureCorrelations(labels,imgNames,c2,c3,organicWnids, ...
    targetWnids,N,outDir,simFile,blockSize,maxSize);
end

inorganicDir = ensureDir([featureDir 'inorganic/']);
if ~exist(inorganicDir,'dir')
  [c3,labels,imageNames] = build5050C3(inorganicC3Files);
  load([inorganicC3Dir 'splits.mat'],'testFiles');
  inorganicWnids = listImageNetCategories(testFiles);
  outFile = [inorganicDir 'pairwiseCorrelations.mat'];
  pairwiseFeatureCorrelations(imgNames,c2,c3,nPairs,maxSize,simFile,outFile);
  categoricalFeatureCorrelations(labels,imgNames,c2,c3,inorganicWnids, ...
    targetWnids,N,outDir,simFile,blockSize,maxSize);
end

