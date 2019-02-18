function types = evaluateFeatureSets(p,basetype,tr_data,te_data,semFile,genFile)
% types = evaluateFeatureSets(p,basetype,tr_data,te_data,semFile,genFile)

    % % % build all the stuff we need in advance % % %
    types = {};

    files_tr= tr_data.file;
    files_te= te_data.file;
    labels_tr = tr_data.label;
    labels_te = te_data.label;
    fprintf('training *and* testing image/label lists configured\n');

    files_tr_ge = cell(length(files_tr),1);
    files_tr_ge2 = cell(length(files_tr),1);
    files_tr_ge3 = cell(length(files_tr),1);
    files_tr_ca = cell(length(files_tr),1);
    files_tr_ca2 = cell(length(files_tr),1);
    parfor iFile = 1:length(files_tr)
        [d,f,~] = fileparts(files_tr{iFile});
        files_tr_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_tr_ge2{iFile} = [d '/' f '.' basetype '_gen2_mat'];
        files_tr_ge3{iFile} = [d '/' f '.' basetype '_gen3_mat'];
        files_tr_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
        files_tr_ca2{iFile} = [d '/' f '.' basetype '_cat2_mat'];
    end
    fprintf('Built training general *and* conceptual feature cache lists\n');

    files_te_ge = cell(length(files_te),1);
    files_te_ca = cell(length(files_te),1);
    parfor iFile = 1:length(files_te)
        [d,f,~] = fileparts(files_te{iFile});
        files_te_ge{iFile} = [d '/' f '.' basetype '_gen_mat'];
        files_te_ge2{iFile} = [d '/' f '.' basetype '_gen2_mat'];
        files_te_ge3{iFile} = [d '/' f '.' basetype '_gen3_mat'];
        files_te_ca{iFile} = [d '/' f '.' basetype '_cat_mat'];
        files_te_ca2{iFile} = [d '/' f '.' basetype '_cat2_mat'];
        % NOTE: ignoring HMAX results.
        % % files_te_hmax_ge{iFile} = [d '/' f '.hmax_gen_mat'];
        % % files_te_hmax_ca{iFile} = [d '/' f '.hmax_cat_mat'];
    end
    fprintf('Built testing general *and* conceptual feature cache lists\n');

    % % % create the functions that actually performs the evaluations % % %

    function multaryEvaluationHelper(outStem,m_tr,m_te,labels_tr,labels_te)

        assert(strcmp(p.method,'logreg'),'EvaluateFeatureSets: method should be logistic regression!\n');

        % create cross-validation splits for training with 1,2,4,8,... examples
        % (testing will always use the same images for comparison)
        splitFile = [p.outDir outStem '-splits.mat'];
        if ~exist(splitFile,'file')
            rngState = rng;
            cvsplit = multiclass_cv(labels_tr,p.nTrainingExamples,p.nRuns);
            save(splitFile,'-mat','rngState','cvsplit');
            fprintf('multiclass splits generated for %s\n',outStem);
        else
            load(splitFile,'-mat','cvsplit');
            fprintf('multiclass splits loaded for %s\n',outStem);
        end

        outFile = [p.outDir outStem '-evaluation.csv'];
        if ~exist(outFile,'file')
            options2 = p.options;
            options2.dir = [options2.dir outStem];
            results = evaluatePerformanceAlt(...
              m_tr',labels_tr',m_te',labels_te',cvsplit,options2);
            writetable(results,outFile);
        end
        fprintf('%s evaluated\n',outStem);

    end

    function [dataFile, outFile] = binaryEvaluationHelper(outStem,tr_file,te_file,nFeatures,score_file,permute)
        if (nargin < 6) permute = false; end;
        if (nargin < 5) score_file = tr_file; end;
        if (nargin < 4), nFeatures = 0; end;

        classFile = [p.outDir 'binary-classes.mat'];
        if ~exist(classFile,'file')
            rngState = rng;
            classes = randperm(size(tr.y,2), p.nBinaryCategories);
            save(classFile,'-mat','-v7.3','rngState','classes');
        else
            load(classFile,'-mat','classes');
        end

        outFile = [p.outDir outStem '-evaluation.csv'];
        eval_dir = ensureDir([p.outDir 'evaluation_data/' outStem '/']);
        dataFile = [eval_dir 'setup.mat'];
        if ~exist(dataFile,'file')
            eval_N = p.options.N;
            nTrain = p.nBinaryTrainingExamples;
            nRuns = p.nRuns;
            thresh = -inf;
            classes = classes;
            save(dataFile, 'tr_file','te_file','score_file','eval_dir', ...
              'eval_N','thresh','nTrain','nFeatures','nRuns','classes','permute');
        end
        fprintf('%s evaluated, %f\n',outStem, posixtime(datetime));
    end

    % % % we'll do the following binary comparisons % % %
    %% - all features
    %% - high mean-activation (thresholded)
    %% - randomly chosen features (thresholded by providing random scores)
    %% - high semantic similarity (based on Wu-Palmer score)
    %% - high visual similarity (only for categorical features)

    % % % create activation matrices for HMAX % % %
    % NOTE: ignoring HMAX results.
    % % buildActivationMatrix(files_te_hmax_ge,labels_te, [p.outDir 'hmax-general-testing-activations.mat']);
    % % buildActivationMatrix(files_te_hmax_ca,labels_te, [p.outDir 'hmax-categorical-testing-activations.mat']);

    % % % evaluate the general features % % %

    te_ge = buildActivationMatrix(files_te_ge,labels_te, [p.outDir 'general-testing-activations.mat']);
    fprintf('Built generic testing activations\n');
    tr_ge = buildActivationMatrix(files_tr_ge,labels_tr, [p.outDir 'general-training-activations.mat']);
    fprintf('Built generic training activations\n');

    % multaryEvaluationHelper([basetype '-general-multary'],m_tr_ge,m_te_ge,labels_tr_ge,labels_te_ge);
    dataFiles = {};
    outFiles = {};
    i = 1;
    [df, of] = binaryEvaluationHelper([basetype '-binary-general'],tr_ge,te_ge);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-general-high-mean-' nFeatures],tr_ge,te_ge,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general-high-semantics-' nFeatures],tr_ge,te_ge,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general-random-' nFeatures],tr_ge,te_ge,nFeatures,tr_ge,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end
    types = [types; {[basetype '-binary-general'];
                     [basetype '-binary-general-high-mean'];
                     [basetype '-binary-general-high-semantics'];
                     [basetype '-binary-general-random']}];

%   % % % evaluate the categorical features % % %

    tr_ca = buildActivationMatrix(files_tr_ca,labels_tr,[p.outDir 'categorical-training-activations.mat']);
    fprintf('Built categorical training activations\n');
    te_ca = buildActivationMatrix(files_te_ca,labels_te,[p.outDir 'categorical-testing-activations.mat']);
    fprintf('Built categorical testing activations\n');

%   multaryEvaluationHelper([basetype '-categorical-multary'],m_tr_ca,m_te_ca,labels_tr_ca,labels_te_ca);
    [df, of] = binaryEvaluationHelper([basetype '-binary-categorical'],tr_ca,te_ca);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical-high-mean-' nFeatures],tr_ca,te_ca,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical-high-semantics-' nFeatures],tr_ca,te_ca,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical-high-generic-' nFeatures],tr_ca,te_ca,nFeatures,genFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical-random-' nFeatures],tr_ca,te_ca,nFeatures,tr_ca,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-categorical'];
                     [basetype '-binary-categorical-high-mean'];
                     [basetype '-binary-categorical-high-semantics'];
                     [basetype '-binary-categorical-high-generic'];
                     [basetype '-binary-categorical-random']}];

%   % % % evaluate the generic 2 features % % %

    te_ge2 = buildActivationMatrix(files_te_ge2,labels_te, [p.outDir 'general2-testing-activations.mat']);
    fprintf('Built generic testing activations\n');
    tr_ge2 = buildActivationMatrix(files_tr_ge2,labels_tr, [p.outDir 'general2-training-activations.mat']);
    fprintf('Built generic training activations\n');

    [df, of] = binaryEvaluationHelper([basetype '-binary-general2'],tr_ge2,te_ge2);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-general2-high-mean-' nFeatures],tr_ge2,te_ge2,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general2-high-semantics-' nFeatures],tr_ge2,te_ge2,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general2-random-' nFeatures],tr_ge2,te_ge2,nFeatures,tr_ge2,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-general2'];
                     [basetype '-binary-general2-high-mean'];
                     [basetype '-binary-general2-high-semantics'];
                     [basetype '-binary-general2-random']}];

%   % % % evaluate the generic 3 features % % %

    te_ge3 = buildActivationMatrix(files_te_ge3,labels_te, [p.outDir 'general3-testing-activations.mat']);
    fprintf('Built generic testing activations\n');
    tr_ge3 = buildActivationMatrix(files_tr_ge3,labels_tr, [p.outDir 'general3-training-activations.mat']);
    fprintf('Built generic training activations\n');

    [df, of] = binaryEvaluationHelper([basetype '-binary-general3'],tr_ge3,te_ge3);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-general3-high-mean-' nFeatures],tr_ge3,te_ge3,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general3-high-semantics-' nFeatures],tr_ge3,te_ge3,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-general3-random-' nFeatures],tr_ge3,te_ge3,nFeatures,tr_ge3,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-general3'];
                     [basetype '-binary-general3-high-mean'];
                     [basetype '-binary-general3-high-semantics'];
                     [basetype '-binary-general3-random']}];

% % % % % evaluate the combined features % % %

    te_co = [p.outDir 'combined-testing-activations.mat'];
    fprintf('Built combined testing activations\n');
    tr_co = [p.outDir 'combined-training-activations.mat'];
    fprintf('Built combined training activations\n');

    [df, of] = binaryEvaluationHelper([basetype '-binary-combined'],tr_co,te_co);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined-high-mean-' nFeatures],tr_co,te_co,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined-high-semantics-' nFeatures],tr_co,te_co,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined-random-' nFeatures],tr_co,te_co,nFeatures,tr_co,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-combined'];
                     [basetype '-binary-combined-high-mean'];
                     [basetype '-binary-combined-high-semantics'];
                     [basetype '-binary-combined-random']}];

%   % % % evaluate the categorical2 features % % %

    tr_ca2 = buildActivationMatrix(files_tr_ca2,labels_tr,[p.outDir 'categorical2-training-activations.mat']);
    fprintf('Built categorical2 training activations\n');
    te_ca2 = buildActivationMatrix(files_te_ca2,labels_te,[p.outDir 'categorical2-testing-activations.mat']);
    fprintf('Built categorical2 testing activations\n');

    [df, of] = binaryEvaluationHelper([basetype '-binary-categorical2'],tr_ca2,te_ca2);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical2-high-mean-' nFeatures],tr_ca2,te_ca2,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical2-high-semantics-' nFeatures],tr_ca2,te_ca2,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical2-high-generic-' nFeatures],tr_ca2,te_ca2,nFeatures,genFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-categorical2-random-' nFeatures],tr_ca2,te_ca2,nFeatures,tr_ca2,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-categorical2'];
                     [basetype '-binary-categorical2-high-mean'];
                     [basetype '-binary-categorical2-high-semantics'];
                     [basetype '-binary-categorical2-high-generic'];
                     [basetype '-binary-categorical2-random']}];

% % % % % evaluate the combined2 features % % %

    te_co = [p.outDir 'combined2-testing-activations.mat'];
    fprintf('Built combined2 testing activations\n');
    tr_co = [p.outDir 'combined2-training-activations.mat'];
    fprintf('Built combined2 training activations\n');

    [df, of] = binaryEvaluationHelper([basetype '-binary-combined2'],tr_co,te_co);
    dataFiles{i} = df;
    outFiles{i} = of;
    i = i+1;
    for iThresh = 1:length(p.testingThreshes)
        nFeatures = num2str(p.testingThreshes(iThresh));
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined2-high-mean-' nFeatures],tr_co,te_co,nFeatures);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined2-high-semantics-' nFeatures],tr_co,te_co,nFeatures,semFile);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
        [df, of] = binaryEvaluationHelper([basetype '-binary-combined2-random-' nFeatures],tr_co,te_co,nFeatures,tr_co,true);
        dataFiles{i} = df;
        outFiles{i} = of;
        i = i+1;
    end

    types = [types; {[basetype '-binary-combined2'];
                     [basetype '-binary-combined2-high-mean'];
                     [basetype '-binary-combined2-high-semantics'];
                     [basetype '-binary-combined2-random']}];

% % % Build the final table % % %

    evalTable = table(dataFiles',outFiles','VariableNames', {'input','output'});
    writetable(evalTable, [p.outDir 'binary_evaluation_input_output_files.csv'], 'WriteVariableNames', 0);

%   % % % evaluate the combination of general and categorical features % % %
% % [m_tr_ge,labels_tr_ge] = buildActivationMatrix(files_tr_ge,labels_tr);
% % [m_te_ge,labels_te_ge] = buildActivationMatrix(files_te_ge,labels_te);
% % fprintf('Built generic activations\n');
% % [m_tr_ca,~] = buildActivationMatrix(files_tr_ca,labels_tr);
% % [m_te_ca,~] = buildActivationMatrix(files_te_ca,labels_te);
% % fprintf('Built categorical activations\n');
% % multaryEvaluationHelper([basetype '-super'],[m_tr_ge;m_tr_ca],[m_te_ge;m_te_ca],labels_tr_ge,labels_te_ge);
% % binaryEvaluationHelper([basetype '-super'],[m_tr_ge;m_tr_ca],[m_te_ge;m_te_ca]);
end
