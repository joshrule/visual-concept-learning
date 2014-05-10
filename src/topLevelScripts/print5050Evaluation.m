function print5050Evaluation(evalDir,metric)
% print5050Evaluation(evalDir,metric)
    evalFiles = listEvaluationFiles(evalDir);
    summary = summarizeEvaluations(evalDir,evalFiles,metric);
    printTable(evalFiles,summary);
end

function evaluationFiles = listEvaluationFiles(evalDir)
    evalFileStructs = dir([evalDir '*evaluation.mat']);
    evaluationFiles = {evalFileStructs.name}';
end

function summary = summarizeEvaluations(evalDir,evalFiles,metric)
    summary = [];
    for iFile = 1:length(evalFiles)
    	evalPath = [evalDir evalFiles{iFile}];
	loadedMetric = load(evalPath,metric);
        summary = [summary; mean(mean(getfield(loadedMetric,metric),3))];
    end
end

function printTable(heading, summary, sep)
    if (nargin < 3), sep = '|'; end;
    maxLength = 0;
    for iFile = 1:length(heading)
        maxLength = max(maxLength,length(heading{iFile}));
    end
    for iFile = 1:length(heading)
        fprintf('%s %s',sprintf('%-*s', maxLength, heading{iFile}), sep);
        fprintf(sprintf(' %%.2f %s',sep),summary(iFile,:));
        fprintf('\n');
    end
end
