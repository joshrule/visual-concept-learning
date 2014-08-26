function c3Simulation(pHandle)
% c3Simulation(pHandle)
%
% Josh Rule <rule@mit.edu>, August 2014
% run the C3 semantics simulations
%
% pHandle, function handle, function loading the parameters
    % initialize environment
    [home,p] = pHandle();
    fprintf('Params Initialized\n\n');

    addpath(genpath([home 'src/']));
    fprintf('Source Loaded\n\n');

    rng(p.seed,'twister');
    fprintf('Pseudorandom Number Generator Reset\n');

    % cache C2 activations
    cacheC2Wrapper(p);

    % build C3 models
    createC3Units(p.organicC3Dir,p.organicC3Params);
    createC3Units(p.inorganicC3Dir,p.inorganicC3Params);

    % choose test categories
    [organicCategories,inorganicCategories,organicC2Files,inorganicC2Files]= ...
        chooseTestCategories(p);
    c2Files = [reshape(organicC2Files,[],1) reshape(inorganicC2Files,[],1)];

    % cache C3 activations
    organicC3Files = regexprep(c2Files,'kmeans.c2',['organic' p.suffix '.c3']);
    cacheC3Wrapper(organicC3Files,c2Files,p.organicC3Dir);

    inorganicC3Files = regexprep(c2Files,'kmeans.c2',['inorganic' p.suffix '.c3']);
    cacheC3Wrapper(inorganicC3Files,c2Files,p.inorganicC3Dir);

    % evaluate performance of models
    evaluteFeatureSets(p,c2Files,organicC3Files,inorganicC3Files);

    % run semantic analysis
    semanticAnalysis(p,c2Files,organicC3Files,inorganicC3Files)
end
