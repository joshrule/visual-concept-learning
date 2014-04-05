home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

hmaxHome = [home 'src/hmax-ocv/'];
kmeansPatchFile = [home 'patchSets/kmeans.xml'];
randomPatchFile = [home 'patchSets/random.xml'];
randomTwoThirdsPatchFile = [home 'patchSets/randomTwoThirds.xml'];
maxSize = 240;
imgDir = [home 'imageSets/animalNoAnimal/'];

animalDir = [imgDir 'animals/'];
animals = dir([animalDir '*.jpg']);
animalImgs = strcat(animalDir,{animals.name});
kmeansAnimalFile = [imgDir 'c2Cache/animals.kmeans.c2.mat'];
randomAnimalFile = [imgDir 'c2Cache/animals.random.c2.mat'];
randomTwoThirdsAnimalFile = [imgDir 'c2Cache/animals.random23.c2.mat'];
if ~exist(kmeansAnimalFile,'file')
    cacheC2(kmeansAnimalFile,kmeansPatchFile,maxSize,animalImgs,hmaxHome);
end
if ~exist(randomAnimalFile,'file')
    cacheC2(randomAnimalFile,randomPatchFile,maxSize,animalImgs,hmaxHome);
end
if ~exist(randomTwoThirdsAnimalFile,'file')
    cacheC2(randomTwoThirdsAnimalFile,randomTwoThirdsPatchFile,maxSize,animalImgs,hmaxHome);
end

noAnimalDir = [imgDir 'noAnimals/'];
noAnimals = dir([noAnimalDir '*.jpg']);
noAnimalImgs = strcat(noAnimalDir,{noAnimals.name});
kmeansNoAnimalFile = [imgDir 'c2Cache/noAnimals.kmeans.c2.mat'];
randomNoAnimalFile = [imgDir 'c2Cache/noAnimals.random.c2.mat'];
randomTwoThirdsNoAnimalFile = [imgDir 'c2Cache/noAnimals.random23.c2.mat'];
if ~exist(kmeansNoAnimalFile,'file')
    cacheC2(kmeansNoAnimalFile,kmeansPatchFile,maxSize,noAnimalImgs,hmaxHome);
end
if ~exist(randomNoAnimalFile,'file')
    cacheC2(randomNoAnimalFile,randomPatchFile,maxSize,noAnimalImgs,hmaxHome);
end
if ~exist(randomTwoThirdsNoAnimalFile,'file')
    cacheC2(randomTwoThirdsNoAnimalFile,randomTwoThirdsPatchFile,maxSize,noAnimalImgs,hmaxHome);
end
