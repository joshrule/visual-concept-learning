function simulation(p)
% simulation(p)
%
% Josh Rule <rule@mit.edu>, March 2016
% run the categorical feature simulations
%
% p, struct, the parameters

    % Save library paths
    MatlabPath = getenv('LD_LIBRARY_PATH');
    % Make Matlab use system libraries
    setenv('LD_LIBRARY_PATH',getenv('PATH'))
%   system( 'R' )
%   % Reassign old library paths
%   setenv('LD_LIBRARY_PATH',MatlabPath)

    status()
    status('It begins')

    cluster = parcluster('local');
    cluster.NumWorkers=32;
    parpool(cluster,32);
    status('initialized 32 thread parallel pool');

    for iPath = 1:length(p.srcPaths)
        addpath(genpath(p.srcPaths{iPath}));
        fprintf('\tsourcing %s...\n',p.srcPaths{iPath});
    end
    status('Sources Loaded');

    rng(p.seed,'twister');
    status('Pseudorandom Number Generator Reset');

    trainingCategories = [p.outDir 'trainingCategories.csv'];
    validationCategories = [p.outDir 'validationCategories.csv'];
    if ~(exist(trainingCategories,'file') && exist(validationCategories,'file'))
        [trCats,evCats] = chooseCategories(p.imgNetDir,p.allCategoryFile,p.validationCategoryFile,p.nImgs,p.nCategories,p.imgNetUser,p.imgNetKey,p.srFile);
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

    trainingImages = [p.outDir,'trainingImages.csv'];
    if ~exist(trainingImages,'file')
        trImages = chooseTrainingImages(trCats.synset,p.imgNetDir,p.nTrValidationImgs); 
        writetable(trImages,trainingImages);
        status('training images divided into training and validation images');
    else
        trImages = readtable(trainingImages);
        status('using pre-computed training image splits');
    end

    validationImages = [p.outDir,'validationImages.csv'];
    if ~exist(validationImages,'file')
        vaImages = chooseValidationImages(p.imgNetValDir);
        writetable(vaImages,validationImages);
        status('evaluation images divided into training and validation images (via ILSVRC2015)');
    else
        vaImages = readtable(validationImages);
        status('using pre-computed validation image splits');
    end

    cacheC2Wrapper(trImages,p.featDir,p.patchFiles,p.hmaxHome,p.maxSize);
    status('cached HMAX c2 activations for training images');

    cacheC2Wrapper(vaImages,p.featDir,p.patchFiles,p.hmaxHome,p.maxSize);
    status('cached HMAX c2 activations for validation images');

%   % do what it says
%   fprintf('models trained\n');

%   % do what it says
%   fprintf('models validated with training validation images\n');
%   
%   % do what it says
%   fprintf('category-general and category-specific features cached for all\n');
%   fprintf('    validation images\n');
%   
%   % do what it says for cat-general, cat-specific and semantic similarity
%   fprintf('visual and semantic similarities computed\n');

%   % evaluate the various feature sets
%   % evaluateFeatureSets(p,c2Files,organicC3Files,inorganicC3Files);
%   fprintf('%s Evaluation Complete\n\n', patchSet);

end

%function splits = chooseSplits(N,ns)
%    if sum(ns) <= N
%        idxs = randperm(N);
%        for i = 1:length(ns)
%            if i == 1
%                splits{i} = idxs(1:ns(1));
%            else
%                splits{i} = idxs(sum(ns(1:(i-1)):sum(ns(1:i))));
%            end
%        end
%    else
%        splits = [];
%    end
%end
%
%function catStruct = chooseCategoriesAndSplits(p,outFile,catFile)
%    if ~exist(outFile,'file')
%        [categories,imgLists] = chooseCategories(outFile,p.imgDir,catFile,p.nImgs,{},p.imgExts);
%        splits = chooseSplits(length(categories),[p.organicC3Params.nModels p.nTestingCategories]);
%        c3Categories = categories(splits{1});
%        c2Categories = categories(splits{2});
%        unusedCategories = categories(splits{3});
%        save(outFile,'rngState','categories','imgLists','splits', ...
%            'c3Categories','c2Categories','unusedCategories','-v7.3');
%        fprintf('%s created\n',outFile);
%    end
%    catStruct = load(outFile,'splits','categories','imgLists','c3Categories','c2Categories');
%    assert(sum(ismember(catStruct.c3Categories,catStruct.c2Categories)) == 0, ...
%           'C3 vocabulary and Test Set overlap!');
%    fprintf('%s loaded\n',outFile);
%end

function status(str)
    if (nargin == 1), fprintf('%s\n',str); end;
    fprintf('========================================')
    fprintf('========================================\n')
end
