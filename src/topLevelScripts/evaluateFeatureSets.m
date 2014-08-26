function evaluateFeatureSets(p,c2Files,organicC3Files,inorganicC3Files)
% c3Evaluation(p,c2Files,organicC3Files,inorganicC3Files)
    assert(isequal(listImageNetCategories(c2Files), ...
                   listImageNetCategories(organicC3Files)) && ...
           isequal(listImageNetCategories(c2Files), ...
                   listImageNetCategories(inorganicC3Files)), ...
           'evaluation: category listing failure');

    % load activations
    [c2,c2Labels] = buildC2(c2Files);
    [oc3,oc3Labels] = buildC3(organicC3Files);
    [ic3,ic3Labels] = buildC3(inorganicC3Files);
    fprintf('Built C2, organic C3, and inorganic C3 activations\n');

    assert(isequal(c2Labels,oc3Labels) && isequal(c2Labels,ic3Labels), ...
        'evaluation: cache failure!');
    labels = c2labels;

    % create cross-validation splits
    if ~exist([outDir 'splits.mat'],'file')
        rngState = rng;
        cvsplit = cv(labels,p.nTrainingExamples,p.nRuns);
        save([outDir 'splits.mat'],'rngState','cvsplit');
        fprintf('50/50 splits generated\n\n');
    else
        load([outDir 'splits.mat'],'cvsplit');
        fprintf('50/50 splits loaded\n\n');
    end

    function c3EvaluationHelper(outStem,m)
        outFile = [outDir outStem '-evaluation.mat'];
        if ~exist(outFile,'file')
            rngState = rng;
            [aucs,dprimes,models,classVals] = evaluatePerformance(m,labels,cvsplit,p.method, ...
              p.options,size(m,1),[]);
            save(outFile,'rngState','labels','m','p','aucs', ...
              'dprimes', 'models', 'classVals', '-v7.3');
            clear rngState m labels aucs dprimes models classVals;
        end
        fprintf('%s evaluated\n',outStem);
    end

    c3EvaluationHelper('kmeans',c2);
    c3EvaluationHelper('organic',oc3);
    c3EvaluationHelper('organic-super',[oc3;c2]);
    c3EvaluationHelper('inorganic',ic3);
    c3EvaluationHelper('inorganic-super',[ic3;c2]);
    c3EvaluationHelper('combined',[oc3;ic3]);
    c3EvaluationHelper('combined-super',[oc3;ic3;c2]);
end
