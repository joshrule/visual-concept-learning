function c3Simulation(pHandle)
% c3Simulation(pHandle)
%
% Josh Rule <rule@mit.edu>, August 2014
% run the C3 semantics simulations
%
% pHandle, function handle, function loading the parameters
    p = pHandle();
    fprintf('Parameters Initialized\n\n');

    addpath(genpath(p.srcPath));
    fprintf('Source Loaded\n\n');

    rng(p.seed,'twister');
    fprintf('Pseudorandom Number Generator Reset\n\n');

    % choose categories and build lists of unique images
    % also split lists of categories into those for C3 models, those for
    % testing, and unused categories
    outFile = [p.outDir 'organic-categories.mat'];
    catFile = [p.imgDir 'organic-categories.mat'];
    organicCategories = chooseCategoriesAndSplits(p,outFile,catFile);

    outFile = [p.outDir 'inorganic-categories.mat'];
    catFile = [p.imgDir 'inorganic-categories.mat'];
    inorganicCategories = chooseCategoriesAndSplits(p,outFile,catFile);
    fprintf('\n');
    
    c2Categories = [reshape(organicCategories.c2Categories,[],1); ...
        reshape(inorganicCategories.c2Categories,[],1)];
    c2ImgLists = {organicCategories.imgLists{organicCategories.splits{2}}  ...
        inorganicCategories.imgLists{inorganicCategories.splits{2}}};

    c3Categories = [reshape(organicCategories.c3Categories,[],1); ...
        reshape(inorganicCategories.c3Categories,[],1)];
    c3ImgLists = {organicCategories.imgLists{organicCategories.splits{1}}  ...
        inorganicCategories.imgLists{inorganicCategories.splits{1}}};

    for iSet = 1:length(p.c2PatchSets)

        % using image lists from above, cache the activations
        patchSet = p.c2PatchSets{iSet};
        cacheImageNetC2(p.caching,patchSet,c2ImgLists,c2Categories);
        fprintf('%s C2 Caching of Test Categories Complete\n\n',patchSet);

        c2Files = strcat(p.caching.cacheDir,c2Categories,'.',patchSet,'.c2.mat');

        if ismember(patchSet,p.c3PatchSets)

            % create C3 units and cache responses
            cacheImageNetC2(p.caching,patchSet,c3ImgLists,c3Categories);
            fprintf('%s C2 Caching of Vocabulary Categories Complete\n\n',patchSet);

            organicC3Dir = [p.home 'patchSets/' patchSet '-organicC3v' p.suffix '/'];
            createC3Units(organicC3Dir,p.organicC3Params,organicCategories.c3Categories,patchSet);
            organicC3Files = strcat(p.caching.cacheDir,c2Categories,'.',patchSet,'-organic',p.suffix,'.c3.mat');
            cacheC3Wrapper(organicC3Files,c2Files,organicC3Dir);
            fprintf('%s Organic C3 Caching Complete\n\n', patchSet);

            inorganicC3Dir = [p.home 'patchSets/' patchSet '-inorganicC3v' p.suffix '/'];
            createC3Units(inorganicC3Dir,p.inorganicC3Params,inorganicCategories.c3Categories,patchSet);
            inorganicC3Files = strcat(p.caching.cacheDir,c2Categories,'.',patchSet,'-inorganic',p.suffix,'.c3.mat');
            cacheC3Wrapper(inorganicC3Files,c2Files,inorganicC3Dir);
            fprintf('%s Inorganic C3 Caching Complete\n\n', patchSet);

            setupSemanticSimilarities(p,c2Files,organicC3Files,inorganicC3Files);
            fprintf('%s Semantic Similarities Complete\n\n', patchSet);
            setupVisualSimilarities(p,patchSet,c2Files);
            fprintf('%s Visual Similarities Complete\n\n', patchSet);
            setupC3Similarities(p,organicC3Files,inorganicC3Files);
            fprintf('%s C3 Similarities Configured\n\n', patchSet);

            % evaluate the various feature sets
            evaluateFeatureSets(p,c2Files,organicC3Files,inorganicC3Files);
            fprintf('%s Evaluation Complete\n\n', patchSet);

            % compute the semantic analysis
            semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files);
            fprintf('%s Semantic Analysis Complete\n\n', patchSet);

        else

            evaluateFeatureSets(p,c2Files);
            fprintf('%s Evaluation Complete\n\n', patchSet);

        end
    end
end

function cacheImageNetC2(p,patchSet,imgs,categories)
    patchFile = [p.patchDir patchSet '.xml'];
    for i = 1:length(categories)
        cacheFile = [p.cacheDir categories{i} '.' patchSet '.c2.mat'];
        if ~exist(cacheFile,'file')
            cacheC2(cacheFile,patchFile,p.maxSize,imgs{i},p.hmaxHome);
        end
    end
end

function splits = chooseSplits(N,ns)
    if sum(ns) <= N
        idxs = randperm(N);
        for i = 1:length(ns)
            if i == 1
                splits{i} = idxs(1:ns(1));
            else
                splits{i} = idxs(sum(ns(1:(i-1)):sum(ns(1:i))));
            end
        end
    else
        splits = [];
    end
end

function catStruct = chooseCategoriesAndSplits(p,outFile,catFile)
    if ~exist(outFile,'file')
        [categories,imgLists] = chooseCategories(outFile,p.imgDir,catFile,p.nImgs,{},p.imgExts);
        splits = chooseSplits(length(categories),[p.organicC3Params.nModels p.nTestingCategories]);
        c3Categories = categories(splits{1});
        c2Categories = categories(splits{2});
        unusedCategories = categories(splits{3});
        save(outFile,'rngState','categories','imgLists','splits', ...
            'c3Categories','c2Categories','unusedCategories','-v7.3');
        fprintf('%s created\n',outFile);
    end
    catStruct = load(outFile,'splits','categories','imgLists','c3Categories','c2Categories');
    assert(sum(ismember(catStruct.c3Categories,catStruct.c2Categories)) == 0, ...
           'C3 vocabulary and Test Set overlap!');
    fprintf('%s loaded\n',outFile);
end
