function validateFeatureSets(p,type,splits)
    outFile = [p.outDir type '-validation.mat'];
    if ~exist(outFile,'file');
        dirInfo = dir([p.outDir type '*-evaluation.mat']);
        candidateFiles = strcat(p.outDir,{dirInfo.name});
        candidateTypes = regexprep(candidateFiles,[p.outDir type '-(?<n>.+)-evaluation\.mat'],'$<n>');
        for iType = 1:length(candidateTypes)
            load(candidateFiles{iType},'dprimes','features');
            for iSplit = 1:size(splits,1)
                trainingData(iSplit,iType,:,:,:) = dprimes(splits(iSplit,:),:,:);
                invSplit = setdiff(1:size(dprimes,1),splits(iSplit,:));
                possibleTestData(iSplit,iType,:,:,:) = dprimes(invSplit,:,:);
                for iTrain = 1:size(trainingData,4)
                    for iSplit2 = 1:size(trainingData,5) % n random splits
                        for iSplit3 = 1:length(invSplit)
                            possibleTestFeatures{iSplit,iType,iSplit3,iTrain,iSplit2} = features{invSplit(iSplit3),iTrain,iSplit2};
                            possibleTestFeatureSizes(iSplit,iType,iSplit3,iTrain,iSplit2) = length(possibleTestFeatures{iSplit,iType,iSplit3,iTrain,iSplit2});
                        end
                    end
                end
            end
        end
        % trainingData Dims = [validationSplits,thresholds,categories,nTrainingExamples,random splits]
        % collapse across categories collected
        collapsedData = squeeze(mean(trainingData,3));
        % top performance for each combination of validationSplit, and nTrainingExamples, taken over thresholds
        for iSplit = 1:size(collapsedData,1)
            for iTrain = 1:size(collapsedData,3)
                for iSplit2 = 1:size(collapsedData,4)
                    [topPerformance(iSplit,iTrain,iSplit2),topIdx(iSplit,iTrain,iSplit2)] = max(collapsedData(iSplit,:,iTrain,iSplit2));
                    chosenCandidate{iSplit,iTrain,iSplit2} = candidateTypes{topIdx(iSplit,iTrain,iSplit2)};
                    load(candidateFiles{topIdx(iSplit,iTrain,iSplit2)},'dprimes','features');
                    invSplit = setdiff(1:size(dprimes,1),splits(iSplit,:));
                    testData(iSplit,:,iTrain,iSplit2) = dprimes(invSplit,iTrain,iSplit2);
                    for iSplit3 = 1:length(invSplit)
                        testFeatures{iSplit,iSplit3,iTrain,iSplit2} = features{invSplit(iSplit3),iTrain,iSplit2};
                        testFeatureSizes(iSplit,iSplit3,iTrain,iSplit2) = length(testFeatures{iSplit,iSplit3,iTrain,iSplit2});
                    end
                end
            end
        end
        save([p.outDir type '-validation.mat'],'candidateFiles','candidateTypes',...
            'trainingData','collapsedData','topPerformance','topIdx', ...
            'chosenCandidate','testData','testFeatures','testFeatureSizes','possibleTestData','possibleTestFeatures','possibleTestFeatureSizes','-v7.3');
    end
    fprintf('%s validation complete\n',type);
end
