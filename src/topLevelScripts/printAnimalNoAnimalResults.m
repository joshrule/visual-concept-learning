function print5050Results(evalDir)
    dirContents = dir([evalDir '*evaluation.mat']);
    evaluationFiles = strcat(evalDir, {dirContents.name});
    for i = 1:length(evaluationFiles)
    	data{i} = load(evaluationFiles{i},'dprimes','aucs');
        [fa,fb{i},fc] = fileparts(evaluationFiles{i});
    end
    fb = regexprep( ...
           regexprep( ...
	     regexprep(fb, ...
	       'plus','+'), ...
	   '-evaluation',''), ...
	 '(\w{1})[a-zA-Z]+','$1');

    load([evalDir 'splits.mat'],'nTrainingExamples');

    dataFn = @(a,b) mean(mean(getfield(a,b),3),1); 
    printTable(data,fb,nTrainingExamples, @(x) dataFn(x,'aucs'));
    printTable(data,fb,nTrainingExamples, @(x) dataFn(x,'dprimes'));
end

function printTable(d,e,n,f)
    fprintf('n\t|'); fprintf(' %d\t|',n); fprintf('\n');
    for i = 1:length(d)
        fprintf('%s\t|',e{i}(1:min(length(e{i}),20))); fprintf(' %.2f\t|',f(d{i})); fprintf('\n');
    end
end
