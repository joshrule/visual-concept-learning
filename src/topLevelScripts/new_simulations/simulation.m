function simulation(p)
% simulation(p)
%
% Josh Rule <rule@mit.edu>, March 2016
% run the categorical feature simulations
%
% p, struct, the parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    status();
    start_dir = pwd;
    status('It begins');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    MatlabPath = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH','/usr/lib/:/usr/local/lib/:/usr/local/cuda/lib:/usr/local/cuda/lib64');
    setenv('CUDA_HOME','/usr/local/cuda');
    status('LD_LIBRARY_PATH updated');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cluster = parcluster('local');
    cluster.NumWorkers=32;
    poolobj = parpool(cluster,32,'IdleTimeout',Inf);
    status(sprintf('initialized %d thread parallel pool',poolobj.NumWorkers));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for iPath = 1:length(p.srcPaths)
        addpath(genpath(p.srcPaths{iPath}));
        fprintf('\tsourcing %s...\n',p.srcPaths{iPath});
    end
    status('Sources Loaded');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    rng(p.seed,'twister');
    status('Pseudorandom Number Generator Reset');

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
        status('Categories chosen for training and evaluating the models');
    else
        trCats = readtable(trainingCategories);
        evCats = readtable(validationCategories);
        status('using pre-computed category choices for training and evaluating the models');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    trainingImages = [p.outDir,'trainingImages.csv'];
    if ~exist(trainingImages,'file')
        trImages = chooseTrainingImages(trCats.synset,p.imgNetDir,p.nTrValidationImgs); 
        writetable(trImages,trainingImages);
        status('training images divided into training and validation images');
    else
        trImages = readtable(trainingImages);
        status('using pre-computed training image splits');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    validationImages = [p.outDir,'validationImages.csv'];
    if ~exist(validationImages,'file')
        vaImages = chooseValidationImages(p.imgNetValDir);
        writetable(vaImages,validationImages);
        status('evaluation images divided into training and validation images (via ILSVRC2015)');
    else
        vaImages = readtable(validationImages);
        status('using pre-computed validation image splits');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % ignoring HMAX results for now!
%%% cacheC2Wrapper(trImages,'hmax_gen',p.featDir,p.patchFiles,p.hmaxHome,p.maxSize);
%%% status('cached HMAX c2 activations for training images');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % ignoring HMAX results for now!
%%% cacheC2Wrapper(vaImages,'hmax_gen',p.featDir,p.patchFiles,p.hmaxHome,p.maxSize);
%%% status('cached HMAX c2 activations for evaluation images');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    prep_lmdb_files([p.home 'caffe/'],p.outDir);
    system('./make_lmdb_files.sh'); % resizes imgs, finds means, makes lmdb DBs
    status('setup LMDB databases');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    r = '';
    while ~strcmp(r,'y')
        r = input('Are the DNN prototxts'' mean values correct (y/n)? ','s');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % ignoring HMAX results for now!
% % system('python make_hmax_lmdb_files.py');
    trainModels(p.caffe_dir);
    status('models trained and evaluated with validation images');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cd(start_dir);
% % % ignoring HMAX results for now!
% % system('python extract_features_hmax.py');
    system('python extract_features_googlenet.py');
    status('general and categorical/conceptual features cached');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    tr_data = readtable([p.home 'caffe/evaluation_training_images.txt'],'Delimiter','space','ReadVariableNames',false);
    tr_data.Properties.VariableNames{'Var1'} = 'file';
    tr_data.Properties.VariableNames{'Var2'} = 'label';
    te_data = readtable([p.home 'caffe/evaluation_validation_images.txt'],'Delimiter','space','ReadVariableNames',false);
    te_data.Properties.VariableNames{'Var1'} = 'file';
    te_data.Properties.VariableNames{'Var2'} = 'label';
    evaluateFeatureSets(p,'googlenet', tr_data, te_data);
    status('Evaluation Complete!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    status('Simulation Complete!');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function status(str)
    if (nargin == 1), fprintf('%s\n',str); end;
    fprintf('========================================')
    fprintf('========================================\n')
end
