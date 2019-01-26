function validateFeatureSets(p,types)
    for type = 1:length(types)
        fprintf('validating results for %s\n', types{type});
        validateFeatureSet(p,types{type});
    end
end

function validateFeatureSet(p,type)
    % if our target file doesn't exist
    outFile = [p.outDir type '-evaluation.csv'];
    if ~exist(outFile,'file');
        % create a list of the relevant evaluations and the feature which varies
        dirInfo = dir([p.outDir type '-*-evaluation.csv']);
        if length(dirInfo) > 0
             candidateFiles = strcat(p.outDir,{dirInfo.name});
             candidateTypes = regexprep(candidateFiles,[p.outDir type '-(?<n>.+)-evaluation\.csv'],'$<n>');
 
             % create a blank table
             validation = table();
 
             % for each feature set x category x training set size, take the mean over random splits
             for iType = 1:length(candidateTypes)
                 evaluations{iType} = readtable(candidateFiles{iType});
             end
 
             classes = unique(evaluations{1}.class);
             nTrain = unique(evaluations{1}.nTraining);
             for iClass = 1:length(classes)
                 for iTrain = 1:length(nTrain)
                     for iType = 1:length(candidateTypes)
                         % select the top performance for each variation x nTrain x class
                         % over a randomly selected training set (half of the random splits)
                         % and report on the test set.
                         rows = evaluations{iType}.class==classes(iClass) & evaluations{iType}.nTraining==nTrain(iTrain);
                         shuffled_rows = rows(randperm(length(rows)));
                         n_rows_over_two = floor(length(rows)/2);
                         training_rows{iType} = shuffled_rows(1:n_rows_over_two);
                         testing_rows{iType} = shuffled_rows((n_rows_over_two+1):end);
                         training_performance(iType) = mean(evaluations{iType}{training_rows{iType},'dprime'});
                         testing_performance(iType) = mean(evaluations{iType}{testing_rows{iType},'dprime'});
                     end
                     % select the test set from the feature set with the highest mean
                     [~,idx] = max(training_performance);
                     rows = testing_rows{idx};
                     validation = [validation; evaluations{idx}(rows,:)];
                     clear rows idx shuffled_rows n_rows_over_two;
                     clear training_rows testing_rows training_performance testing_performance;
                 end
             end
             writetable(validation,outFile);
        else
            fprintf('already validated!\n');
        end
    end
end
