function data = printEvaluations(evalDir,metric,points,plotP,printP)
% printEvaluations(evalDir,metric)
    if (nargin < 5) printP = 1; end;
    if (nargin < 4) plotP  = 0; end;
    evalFiles = listEvaluationFiles(evalDir);
    [summary,data] = summarizeEvaluations(evalDir,evalFiles,metric);
    if plotP
        plotTable(evalDir,evalFiles,data,points);
    end
    if printP
        printTable(evalFiles,summary);
    end
end

function evaluationFiles = listEvaluationFiles(evalDir)
    evalFileStructs = dir([evalDir '*evaluation.mat']);
    evaluationFiles = {evalFileStructs.name}';
end

function [summary,data] = summarizeEvaluations(evalDir,evalFiles,metric)
    summary = [];
    for iFile = 1:length(evalFiles)
    	evalPath = [evalDir evalFiles{iFile}];
        loadedMetric = load(evalPath,metric);
        summary = [summary; mean(mean(getfield(loadedMetric,metric),3))];
        data(iFile,:,:,:) = getfield(loadedMetric,metric);
    end
end

function plotTable(evalDir,evalFiles,data,points)
   clf;
   hold on;
   colors = {'b','r','g','k','m', 'c','y'};
   for i = 1:length(evalFiles)
       summary = squeeze(mean(mean(data(i,:,:,:),4),2));
       semFeed = shiftdim(squeeze(data(i,:,:,:)),1);
       for iSem = 1:size(semFeed,1)
           sem(iSem) = std(reshape(semFeed(iSem,:,:),[],1))/sqrt(numel(semFeed(iSem,:,:)));
       end
       errorbar(points,summary,sem,colors{(mod(i,length(colors))+1)});
   end
   ylabel('d''');
   xlabel('# training examples (total)');
   title('AnA Evaluation Performance (Linear Kernels)');
   evalNames = regexprep(evalFiles,'-evaluation.mat','');
   legend(evalNames,'Location','SouthEast');
   print(gcf,[evalDir 'evaluations.eps'],'-depsc');
   clf; hold off;
end

function printTable(heading, summary, sep)
    if (nargin < 3), sep = '|'; end;
    maxLength = 0;
    heading = regexprep(heading,'-evaluation.mat','');
    for iFile = 1:length(heading)
        maxLength = max(maxLength,length(heading{iFile}));
    end
    for iFile = 1:length(heading)
        fprintf('%s %s',sprintf('%-*s', maxLength, heading{iFile}), sep);
        fprintf(sprintf(' %%.2f %s',sep),summary(iFile,:));
        fprintf('\n');
    end
end
