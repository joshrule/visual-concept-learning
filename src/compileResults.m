function compileResults(p,type)
    outFile = [p.outDir 'final_results.csv'];
    if ~exist(outFile, 'file')
        results = table();
        dirInfo = dir([p.outDir type '*-evaluation.csv']);
        if length(dirInfo) > 0
            types = {dirInfo.name};
        for type = 1:length(types)
            fprintf('compiling results for %s\n', types{type});
            t = readtable([p.outDir types{type}]);
            t.featureSet = repmat({types{type}}, height(t), 1);
            t = t(:,{'featureSet','class','nTraining','iRun','dprime'});
            t.Properties.VariableNames{'nTraining'} = 'nTrainingExamples';
            t.Properties.VariableNames{'iRun'} = 'split';
            t.Properties.VariableNames{'dprime'} = 'score';
            results = [results; t];
        end
        writetable(results, outFile);
    end
end
