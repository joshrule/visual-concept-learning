home = '/home/josh/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

imgDir = [home 'imageSets/animalNoAnimal/'];
outDir = ensureDir([home 'evaluation/halfNegs/']);

method = 'svm';
options = '-s 0 -t 0 -b 1 -q -c 0.1';
labels = [ones(1,600) zeros(1,600)];

if ~exist([outDir 'splits.mat'],'file')
    nTrainingExamples = [16 32 64 128 256 512 600 1024];
    nRuns = 40;
    cvsplit = cv(labels,nTrainingExamples,nRuns);
    save([outDir 'splits.mat'],'nTrainingExamples','nRuns','cvsplit');
    fprintf('splits generated\n');
else
    load([outDir 'splits.mat'])
    fprintf('splits loaded\n');
end

% k-means 400
if ~exist([outDir 'kmeans-evaluation.mat'],'file')
    animals = load([imgDir 'c2Cache/animals.kmeans.c2.mat'], 'c2');
    noAnimals = load([imgDir 'c2Cache/noAnimals.kmeans.c2.mat'], 'c2');
    c2 = [animals.c2 noAnimals.c2];
    [aucs,dprimes,models] = evaluatePerformance(c2,labels,cvsplit,method, ...
      options,size(c2,1),[]);
    save([outDir 'kmeans-evaluation.mat'],'labels','c2','aucs','dprimes', ...
         'models','-v7.3');
    clear animals noAnimals c2 aucs dprimes models;
end
fprintf('kmeans evaluated\n');

% organic
if ~exist([outDir 'organic-evaluation.mat'],'file')
    animals = load([imgDir 'c2Cache/animals.organicHalfNegs.c3.mat'], 'c3');
    noAnimals = load([imgDir 'c2Cache/noAnimals.organicHalfNegs.c3.mat'], 'c3');
    c3 = [animals.c3 noAnimals.c3];
    [aucs,dprimes,models] = evaluatePerformance(c3,labels,cvsplit,method, ...
      options,size(c3,1),[]);
    save([outDir 'organic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c3 aucs dprimes models;
end
fprintf('organic evaluated\n');

% organic-super
if ~exist([outDir 'organic-super-evaluation.mat'],'file')
    c2Animals = load([imgDir 'c2Cache/animals.kmeans.c2.mat'], 'c2');
    c2NoAnimals = load([imgDir 'c2Cache/noAnimals.kmeans.c2.mat'], 'c2');
    c3Animals = load([imgDir 'c2Cache/animals.organicHalfNegs.c3.mat'], 'c3');
    c3NoAnimals = load([imgDir 'c2Cache/noAnimals.organicHalfNegs.c3.mat'], 'c3');
    c2c3 = [c2Animals.c2 c2NoAnimals.c2; c3Animals.c3 c3NoAnimals.c3];
    [aucs,dprimes,models] = evaluatePerformance(c2c3,labels,cvsplit,method, ...
      options,size(c2c3,1),[]);
    save([outDir 'organic-super-evaluation.mat'],'labels','c2c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c2c3 aucs dprimes models;
end
fprintf('organic-super evaluated\n');

% inorganic
if ~exist([outDir 'inorganic-evaluation.mat'],'file')
    animals = load([imgDir 'c2Cache/animals.inorganicHalfNegs.c3.mat'], 'c3');
    noAnimals = load([imgDir 'c2Cache/noAnimals.inorganicHalfNegs.c3.mat'], 'c3');
    c3 = [animals.c3 noAnimals.c3];
    [aucs,dprimes,models] = evaluatePerformance(c3,labels,cvsplit,method, ...
      options,size(c3,1),[]);
    save([outDir 'inorganic-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c3 aucs dprimes models;
end
fprintf('inorganic evaluated\n');

% inorganic-super
if ~exist([outDir 'inorganic-super-evaluation.mat'],'file')
    c2Animals = load([imgDir 'c2Cache/animals.kmeans.c2.mat'], 'c2');
    c2NoAnimals = load([imgDir 'c2Cache/noAnimals.kmeans.c2.mat'], 'c2');
    c3Animals = load([imgDir 'c2Cache/animals.inorganicHalfNegs.c3.mat'], 'c3');
    c3NoAnimals = load([imgDir 'c2Cache/noAnimals.inorganicHalfNegs.c3.mat'], 'c3');
    c2c3 = [c2Animals.c2 c2NoAnimals.c2; c3Animals.c3 c3NoAnimals.c3];
    [aucs,dprimes,models] = evaluatePerformance(c2c3,labels,cvsplit,method, ...
      options,size(c2c3,1),[]);
    save([outDir 'inorganic-super-evaluation.mat'],'labels','c2c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c2c3 aucs dprimes models;
end
fprintf('inorganic-super evaluated\n');

% combined
if ~exist([outDir 'combined-evaluation.mat'],'file')
    animalsOrganic = load([imgDir 'c2Cache/animals.organicHalfNegs.c3.mat'], 'c3');
    animalsInorganic = load([imgDir 'c2Cache/animals.inorganicHalfNegs.c3.mat'], 'c3');
    noAnimalsOrganic = load([imgDir 'c2Cache/noAnimals.organicHalfNegs.c3.mat'], 'c3');
    noAnimalsInorganic = load([imgDir 'c2Cache/noAnimals.inorganicHalfNegs.c3.mat'], 'c3');
    c3 = [animalsOrganic.c3   noAnimalsOrganic.c3; ...
          animalsInorganic.c3 noAnimalsInorganic.c3];
    [aucs,dprimes,models] = evaluatePerformance(c3,labels,cvsplit,method, ...
      options,size(c3,1),[]);
    save([outDir 'combined-evaluation.mat'],'labels','c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c3 aucs dprimes models;
end
fprintf('combined evaluated\n');

% combined-super
if ~exist([outDir 'combined-super-evaluation.mat'],'file')
    c2Animals = load([imgDir 'c2Cache/animals.kmeans.c2.mat'], 'c2');
    c2NoAnimals = load([imgDir 'c2Cache/noAnimals.kmeans.c2.mat'], 'c2');
    c3AnimalsOrganic = load([imgDir 'c2Cache/animals.organicHalfNegs.c3.mat'], 'c3');
    c3AnimalsInorganic = load([imgDir 'c2Cache/animals.inorganicHalfNegs.c3.mat'], 'c3');
    c3NoAnimalsOrganic = load([imgDir 'c2Cache/noAnimals.organicHalfNegs.c3.mat'], 'c3');
    c3NoAnimalsInorganic = load([imgDir 'c2Cache/noAnimals.inorganicHalfNegs.c3.mat'], 'c3');
    c2c3 = [c2Animals.c2          c2NoAnimals.c2 ; ... 
            c3AnimalsOrganic.c3   c3NoAnimalsOrganic.c3; ...
            c3AnimalsInorganic.c3 c3NoAnimalsInorganic.c3];
    [aucs,dprimes,models] = evaluatePerformance(c2c3,labels,cvsplit,method, ...
      options,size(c2c3,1),[]);
    save([outDir 'combined-super-evaluation.mat'],'labels','c2c3','aucs','dprimes', ...
      'models','-v7.3');
    clear animals noAnimals c2c3 aucs dprimes models;
end
fprintf('combined-super evaluated\n');
