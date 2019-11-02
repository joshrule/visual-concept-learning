function compileResults(p,type)
% compileResults(p,type)
%
% Collect the main evaluation results into a single CSV.
%
% Args:
% - p: struct, the simulation parameters (see `params.m`)
% - type: string, the evaluations being compiled (e.g. 'googlenet-binary')
    % If the results haven't been written yet...
    outFile = [p.outDir 'final_results.csv'];
    if ~exist(outFile, 'file')
        % Create an empty table of results.
        results = table();
        % Figure out which sub-simulations are involved.
        dirInfo = dir([p.outDir type '*-evaluation.csv']);
        if length(dirInfo) > 0
            types = {dirInfo.name};
        % Append the results of each subsimulation to the main table.
        for type = 1:length(types)
            fprintf('compiling results for %s\n', types{type});
            t = readtable([p.outDir types{type}]);
            t.featureSet = repmat({types{type}}, height(t), 1);
            t = t(:,{'featureSet','class','nTraining','iRun','precision','recall','accuracy','pr_auc','roc_auc','F','dprime'});
            t.Properties.VariableNames{'nTraining'} = 'nTrainingExamples';
            t.Properties.VariableNames{'iRun'} = 'split';
            results = [results; t];
        end
        % Save the compiled results to disk.
        writetable(results, outFile);
    end
end
