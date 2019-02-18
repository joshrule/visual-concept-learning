function evaluateFeatureSets(p,basetype,tr_data,te_data,semFile,genFile)
% evaluateFeatureSets(p,basetype,tr_data,te_data,semFile,genFile)

    % % % build all the stuff we need in advance % % %

    featureSets = {'categorical', 'categorical2', 'combined', 'combined2', 'general', 'general2', 'general3'};

    dataFilesE = cell(length(featureSets),1);
    outFilesE = cell(length(featureSets),1);
    dataFilesC = cell(length(featureSets),1);
    outFilesC = cell(length(featureSets),1);

    % % % create the functions that actually performs the evaluations % % %

    % function multaryEvaluationHelper(outStem,m_tr,m_te,labels_tr,labels_te)
    %     % create cross-validation splits for training with 1,2,4,8,... examples
    %     % (testing will always use the same images for comparison)
    %     splitFile = [p.outDir outStem '-splits.mat'];
    %     if ~exist(splitFile,'file')
    %         rngState = rng;
    %         cvsplit = multiclass_cv(labels_tr,p.nTrainingExamples,p.nRuns);
    %         save(splitFile,'-mat','rngState','cvsplit');
    %         fprintf('multiclass splits generated for %s\n',outStem);
    %     else
    %         load(splitFile,'-mat','cvsplit');
    %         fprintf('multiclass splits loaded for %s\n',outStem);
    %     end
    %
    %     outFile = [p.outDir outStem '-evaluation.csv'];
    %     if ~exist(outFile,'file')
    %         options2 = p.options;
    %         options2.dir = [options2.dir outStem];
    %         results = evaluatePerformanceAlt(...
    %           m_tr',labels_tr',m_te',labels_te',cvsplit,options2);
    %         writetable(results,outFile);
    %     end
    %     fprintf('%s evaluated\n',outStem);
    %
    % end

    function [dataFile, outFile] = binaryEvaluationHelper(outStem,tr_file,te_file,nFeatures,score_file,permute)
        if (nargin < 6) permute = false; end;
        if (nargin < 5) score_file = tr_file; end;
        if (nargin < 4), nFeatures = 0; end;

        classFile = [p.outDir 'subset-of-classes.mat'];
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

    function [dataFile, outFile] = categoricityEvaluationHelper(outStem,tr_file,te_file)
        classFile = [p.outDir 'subset-of-classes.mat'];
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
            nTrain = p.nCategoricityTrainingExamples;
            classes = classes;
            save(dataFile, 'tr_file','te_file','eval_dir','nTrain','classes');
        end
        fprintf('%s evaluated, %f\n',outStem, posixtime(datetime));
    end

    function processFeatureSet(label, idx)
        te_co = [p.outDir label '-testing-activations.mat'];
        fprintf(['Built ' label ' testing activations\n']);
        tr_co = [p.outDir label '-training-activations.mat'];
        fprintf(['Built ' label ' training activations\n']);

        % multaryEvaluationHelper([basetype '-' label '-multary'],m_tr_ge,m_te_ge,labels_tr_ge,labels_te_ge);

        [df, of] = binaryEvaluationHelper([basetype '-binary-' label],tr_co,te_co);
        dataFilesE{idx} = df;
        outFilesE{idx} = of;

        [df, of] = categoricityEvaluationHelper([basetype '-categoricity-' label],tr_co,te_co);
        dataFilesC{idx} = df;
        outFilesC{idx} = of;
    end

    % % % setup evaluations for each feature set % % %

    for iFS = 1:length(featureSets)
        processFeatureSet(featureSets{iFS}, iFS);
    end

    % % % Build the input/output tables % % %

    evalTable = table(dataFilesE,outFilesE,'VariableNames', {'input','output'});
    writetable(evalTable, [p.outDir 'binary_evaluation_input_output_files.csv'], 'WriteVariableNames', 0);

    catTable = table(dataFilesC,outFilesC,'VariableNames', {'input','output'});
    writetable(catTable, [p.outDir 'categoricity_evaluation_input_output_files.csv'], 'WriteVariableNames', 0);

end
