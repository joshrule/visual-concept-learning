home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

params = struct( ...
    'home', home, ...
    'trainDir', [home 'imageSets/imageNet/organicC2Cache/'], ...
    'trainN', 1000, ...
    'testDir', [home 'imageSets/animalNoAnimal/c2Cache/'], ...
    'testN', 2, ...
    'patchSet', 'kmeans', ...
    'minPerClass', 150, ...
    'trainOptions', struct('svmTrainFlags', '-s 0 -t 0 -c 0.1 -b 1 -q', ...
                           'svmTestFlags', '-b 1', ...
                           'alpha', 0.7, ...
                           'startPerIter', 200, ...
                           'threshold', 0.25), ...
    'testOptions', '-s 0 -t 0 -c 0.1 -b 1 -q', ...
    'nTrainingExamples', [16 32 64 128 256 512 1024], ...
    'nRuns', 20, ...
    'repRatio', 1, ...
    'mining', false, ...
    'method', 'svm');

c3Simulation([home 'patchSets/organicC3vLinear/'],params);
