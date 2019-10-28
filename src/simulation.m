function simulation(p)
% simulation(p)
%
% Josh Rule <rule@mit.edu>, September 2019
% evaluate concept + generic feature performance
%
% Args: 
%   p: struct, the parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    status();
    status('It begins');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    start_dir = pwd;
    MatlabPath = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH','/usr/lib/:/usr/local/lib/:/usr/local/cuda/lib:/usr/local/cuda/lib64');
    setenv('CUDA_HOME','/usr/local/cuda');

    status('Updated LD_LIBRARY_PATH');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cluster = parcluster('local');
    cluster.NumWorkers=32;
    poolobj = parpool(cluster,32,'IdleTimeout',Inf);

    status(sprintf('Initialized %d thread parallel pool',poolobj.NumWorkers));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for iPath = 1:length(p.srcPaths)
        addpath(genpath(p.srcPaths{iPath}));
        fprintf('sourcing %s...\n',p.srcPaths{iPath});
    end

    status('Loaded sources');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    rng(p.seed,'twister');

    status('Reset pseudorandom number generator');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    trainingCategories = [p.outDir 'trainingCategories.csv'];
    validationCategories = [p.outDir 'validationCategories.csv'];
    if ~(exist(trainingCategories,'file') && exist(validationCategories,'file'))
        [trCats,evCats] = chooseCategories(p.imgNetDir,p.allCategoryFile, ...
          p.validationCategoryFile,p.nImgs,p.nCategories,p.imgNetUser, ...
          p.imgNetKey,p.srFile);
        trCats = table(trCats,'VariableNames',{'synset'});
        evCats = table(evCats,'VariableNames',{'synset'});
        writetable(trCats,trainingCategories);
        writetable(evCats,validationCategories);

        status('Chose categories for training and evaluating the models');
    else
        trCats = readtable(trainingCategories, 'Delimiter', ',');
        evCats = readtable(validationCategories, 'Delimiter', ',');

        status('Loaded pre-computed category choices for training and evaluating the models');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    trainingImages = [p.outDir,'trainingImages.csv'];
    if ~exist(trainingImages,'file')
        trImages = chooseTrainingImages(trCats.synset,p.imgNetDir,p.nTrValidationImgs); 
        writetable(trImages,trainingImages);

        status('Split training images into training and validation images');
    else
        trImages = readtable(trainingImages, 'Delimiter', ',');

        status('Loaded pre-computed training image splits');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    validationImages = [p.outDir,'validationImages.csv'];
    if ~exist(validationImages,'file')
        vaImages = chooseValidationImages(p.imgNetValDir);
        writetable(vaImages,validationImages);

        status('Split evaluation images into training and validation images (via ILSVRC2015)');
    else
        vaImages = readtable(validationImages, 'Delimiter', ',');

        status('Loaded pre-computed validation image splits');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    prepLMDBFiles([p.home 'caffe/'],p.outDir);
    system('./make_lmdb_files.sh'); % resizes imgs, finds means, makes lmdb DBs

    status('Setup LMDB databases');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    r = '';
    while ~strcmp(r,'y')
        r = input('Are the DNN prototxts'' mean values correct (y/n)? ','s');
    end
    trainModels(p.caffe_dir);

    status('Trained models and evaluated on validation images');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cd(start_dir);
    system('python cache_features.py');

    status('Cached generic and concept features');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    semSimFile = cacheSemanticSimilarities( ...
      [p.outDir 'semantic_similarities/'], trCats.synset, vaImages);
    genSimFile = cacheVisualSimilarities( ...
      [p.outDir 'visual_similarities/'], trImages, vaImages, 'googlenet');

    status('Cached similarities!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tr_data = readtable([p.home 'caffe/evaluation_training_images.txt'],'Delimiter','space','ReadVariableNames',false);
    tr_data.Properties.VariableNames{'Var1'} = 'file';
    tr_data.Properties.VariableNames{'Var2'} = 'label';
    te_data = readtable([p.home 'caffe/evaluation_validation_images.txt'],'Delimiter','space','ReadVariableNames',false);
    te_data.Properties.VariableNames{'Var1'} = 'file';
    te_data.Properties.VariableNames{'Var2'} = 'label';
    evaluateFeatureSets(p,'googlenet', tr_data, te_data, semSimFile, genSimFile);

    status('Setup feature set evaluations!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    compileResults(p,'googlenet-binary');

    status('Compiled results!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    status('It ends!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function status(str)
    if (nargin == 1), fprintf('%s\n',str); end;
    fprintf([repmat('=',1,80) '\n']);
end
