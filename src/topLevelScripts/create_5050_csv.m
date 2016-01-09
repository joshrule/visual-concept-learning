function create_5050_csv(pfh,prefix,resDir,outDir)
% create_5050_csv(pfh,resDir,outDir)
p = pfh();
d = [];
vd = [];

oc = load([outDir 'organic-categories.mat'],'c2Categories');
ic = load([outDir 'inorganic-categories.mat'],'c2Categories');

fsTypes = {'universal (k-means)', ...
           'C3', ...
           'universal (k-means) + C3', ...
           '2007', ...
           'low mean-activation universal C2', ...
           'low mean-activation C3', ...
           'high mean-activation C3', ...
           'high visual similarity C3', ...
           'high semantic similarity C3'};
c2Types = [oc.c2Categories; ic.c2Categories];
teTypes = p.nTrainingExamples;
splitTypes = 1:p.nRuns;

load([resDir 'featureSetSplits.mat'],'featureSetSplit');
for i = 1:size(featureSetSplit,1)
    invSplit(i,:) = setdiff(1:length(c2Types),featureSetSplit(i,:));
end

fprintf('starting to create the summary structs\n');
d = add_standard_results(p,d,fsTypes{1},[resDir 'kmeans-evaluation.mat']);
d = add_standard_results(p,d,fsTypes{2},[resDir 'kmeans-combined-evaluation.mat']);
d = add_standard_results(p,d,fsTypes{3},[resDir 'kmeans-combined-super-evaluation.mat']);
d = add_standard_results(p,d,fsTypes{4},[resDir 'twoOhOhSeven-evaluation.mat']);
[d,vd] = add_thresholded_results(p,d,vd,fsTypes{5},[resDir 'kmeans-below-thresh-validation.mat']);
[d,vd] = add_thresholded_results(p,d,vd,fsTypes{6},[resDir 'kmeans-combined-thresh-inverted-validation.mat']);
[d,vd] = add_thresholded_results(p,d,vd,fsTypes{7},[resDir 'kmeans-combined-thresh-normal-validation.mat']);
[d,vd] = add_thresholded_results(p,d,vd,fsTypes{8},[resDir 'kmeans-combined-thresh-by-visual-validation.mat']);
[d,vd] = add_thresholded_results(p,d,vd,fsTypes{9},[resDir 'kmeans-combined-thresh-by-semantics-validation.mat']);

fprintf('starting to create the linearizedData\n');
c = 0;
for iFS = 1:4
    for iCat = 1:length(c2Types)
        for iTrain = 1:length(teTypes)
            for iSplit = 1:length(splitTypes)
                c = c+1;
                linearizedData(c).featureSet = fsTypes{iFS};
                linearizedData(c).category = c2Types{iCat};
                linearizedData(c).nTrainingExamples = teTypes(iTrain);
                linearizedData(c).split = iSplit;
                linearizedData(c).dprime = d(iFS).dprimes(iCat,iTrain,iSplit);
            end
        end
    end
    fprintf('%d\n',iFS);
end
for iFS = 5:9
    for iCat = 1:size(invSplit,2)
        for iTrain = 1:length(teTypes)
            for iSplit = 1:length(splitTypes)
                for iRep = 1:size(invSplit,1)
                    c = c+1;
                    linearizedData(c).featureSet = fsTypes{iFS};
                    linearizedData(c).category = c2Types{invSplit(iRep,iCat)};
                    linearizedData(c).nTrainingExamples = teTypes(iTrain);
                    linearizedData(c).split = iSplit;
                    linearizedData(c).dprime = d(iFS).dprimes(iRep,iCat,iTrain,iSplit);
                end
            end
        end
    end
    fprintf('%d\n',iFS);
end

results.data = d;
clear d;
results.validationData = vd;
clear vd;
results.(['imageNet_' prefix '_struct']) = linearizedData;
clear linearizedData;

fprintf('saving structs to disk\n');
save([outDir 'imageNet_' prefix '_summary_struct.mat'],'-struct','results','validationData','data');
save([outDir 'imageNet_' prefix '_struct.mat'],'-struct','results',['imageNet_' prefix '_struct']);
fprintf('generating table and saving it to disk\n');
t = struct2table(results.(['imageNet_' prefix '_struct']));
writetable(t,[outDir 'imageNet_' prefix '_table.csv']);
end

function [d,vd] = add_thresholded_results(p,d,vd,name,file)
    i = length(d)+1;
    iv = length(vd)+1;
    load(file,'testData','candidateTypes','chosenCandidate','possibleTestFeatureSizes','possibleTestData');
    for iTrain = 1:size(testData,3)
        tmp = reshape(testData(:,:,iTrain,:),[],1);
        d(i).data(iTrain) = mean(tmp);
        d(i).std(iTrain) = std(tmp);
        d(i).n(iTrain) = length(tmp);
    end
    d(i).names = name;
    d(i).nTrainingExamples = p.nTrainingExamples;
    d(i).dprimes = testData;
    vd(iv).names = name;
    vd(iv).possibilities = candidateTypes;
    vd(iv).values = chosenCandidate;
    vd(iv).data = testData;
    vd(iv).allData = possibleTestData;
    vd(iv).allNFeats = possibleTestFeatureSizes;
end

function d = add_standard_results(p,d,name,file)
    i = length(d)+1;
    load(file,'dprimes');
    for iTrain = 1:size(dprimes,2)
        tmp = reshape(dprimes(:,iTrain,:),[],1);
        d(i).data(iTrain) = mean(tmp);
        d(i).std(iTrain) = std(tmp);
        d(i).n(iTrain) = length(tmp);
    end
    d(i).names = name;
    d(i).nTrainingExamples = p.nTrainingExamples;
    d(i).dprimes = dprimes;
end
