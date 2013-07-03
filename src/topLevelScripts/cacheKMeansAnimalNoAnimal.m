home = '/home/joshrule/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

hmaxHome = [home 'src/hmax-ocv/'];
patchFile = [home 'patchSets/kmeans.xml'];
maxSize = 240;
imgDir = [home 'imageSets/animalNoAnimal/'];

animalDir = [imgDir 'animals/'];
animals = dir([animalDir '*.jpg']);
animalImgs = strcat(animalDir,{animals.name});
animalFile = [imgDir 'c2Cache/animals.kmeans.c2.mat'];
cacheC2(animalFile,patchFile,maxSize,animalImgs,hmaxHome);

noAnimalDir = [imgDir 'noAnimals/'];
noAnimals = dir([noAnimalDir '*.jpg']);
noAnimalImgs = strcat(noAnimalDir,{noAnimals.name});
noAnimalCache = [imgDir 'c2Cache/noAnimals.kmeans.c2.mat'];
cacheC2(noAnimalCache,patchFile,maxSize,noAnimalImgs,hmaxHome);
