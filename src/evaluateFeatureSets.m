function evaluateFeatureSets(p,basetype)
% evaluateFeatureSets(p,basetype)
%
% Configure the feature set evaluations.
%
% Args:
% - p: struct, the simulation parameters (see `params.m`)
% - basetype: string, the evaluations being compiled (e.g. 'googlenet-binary')

    % % % Initialize a few useful vectors. % % %

    featureSets = {'categorical2', 'combined2', 'general', 'general2', 'general3'};

    dataFilesE = cell(length(featureSets),1);
    outFilesE = cell(length(featureSets),1);
    dataFilesC = cell(length(featureSets),1);
    outFilesC = cell(length(featureSets),1);

    % % % Create helper functions for configuring the evaluations. % % %

    function [dataFile, outFile] = binaryEvaluationHelper(outStem,tr_file,te_file,small,score_file)
        % Obsolete: create a default score matrix if none exists.
        if (nargin < 5) score_file = tr_file; end;

        % Save/load a list of classes to be used.
        classFile = [p.outDir 'subset-of-classes.mat'];
        if ~exist(classFile,'file')
            rngState = rng;
            classes = randperm(size(tr.y,2), p.nBinaryCategories);
            save(classFile,'-mat','-v7.3','rngState','classes');
        else
            load(classFile,'-mat','classes');
        end

        % Save the evaluation parameters.
        outFile = [p.outDir outStem '-evaluation.csv'];
        eval_dir = ensureDir([p.outDir 'evaluation_data/' outStem '/']);
        dataFile = [eval_dir 'setup.mat'];
        if ~exist(dataFile,'file')
            nTrain = p.nBinaryTrainingExamples;
            nRuns = p.nRuns;
            nFeatures = -1;
            thresh = -inf;
            classes = classes;
            save(dataFile, 'tr_file','te_file','score_file','eval_dir', ...
              'thresh','nTrain','nFeatures','nRuns','classes','small');
        end
        fprintf('%s evaluated, %f\n',outStem, posixtime(datetime));
    end

    function [dataFile, outFile] = categoricityEvaluationHelper(outStem,tr_file,te_file)
        % Save/load a list of classes to be used.
        classFile = [p.outDir 'subset-of-classes.mat'];
        if ~exist(classFile,'file')
            rngState = rng;
            classes = randperm(size(tr.y,2), p.nBinaryCategories);
            save(classFile,'-mat','-v7.3','rngState','classes');
        else
            load(classFile,'-mat','classes');
        end

        % Save the evaluation parameters
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
        % Create filenames containing the necessary activation matrices.
        te_co = [p.outDir label '-testing-activations.mat'];
        tr_co = [p.outDir label '-training-activations.mat'];

        % Collect the model parameters for these 3 feature sets.
        small = ~(strcmp(label,'categorical2') | strcmp(label,'combined2') | strcmp(label,'general'));

        % Configure the binary evaluations.
        [df, of] = binaryEvaluationHelper([basetype '-binary-' label],tr_co,te_co,small);
        dataFilesE{idx} = df;
        outFilesE{idx} = of;

        % Configure the categoricity evaluations.
        [df, of] = categoricityEvaluationHelper([basetype '-categoricity-' label],tr_co,te_co);
        dataFilesC{idx} = df;
        outFilesC{idx} = of;
    end

    % % % Configure evaluations for each feature set. % % %

    for iFS = 1:length(featureSets)
        processFeatureSet(featureSets{iFS}, iFS);
    end

    % % % Build the input/output tables. % % %

    evalTable = table(dataFilesE,outFilesE,'VariableNames', {'input','output'});
    writetable(evalTable, [p.outDir 'binary_evaluation_input_output_files.csv'], 'WriteVariableNames', 0);

    catTable = table(dataFilesC,outFilesC,'VariableNames', {'input','output'});
    writetable(catTable, [p.outDir 'categoricity_evaluation_input_output_files.csv'], 'WriteVariableNames', 0);

end
