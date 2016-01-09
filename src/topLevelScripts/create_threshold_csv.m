function create_threshold_csv(pfh,prefix,resDir,outDir)
% create_threshold_csv(pfh,prefix,resDir,outDir)
p = pfh();

oc = load([resDir 'organic-categories.mat'],'c2Categories');
ic = load([resDir 'inorganic-categories.mat'],'c2Categories');

c2Types = [oc.c2Categories; ic.c2Categories];
teTypes = p.nTrainingExamples;
splitTypes = 1:p.nRuns;
cvTypes = 1:p.nCVRuns;
fsTypes = {'universal (k-means)', ...
           'C3', ...
           'low mean-activation universal C2', ...
           'low mean-activation C3', ...
           'high mean-activation C3', ...
           'high visual similarity C3', ...
           'high semantic similarity C3'};

load([resDir 'featureSetSplits.mat'],'featureSetSplit');
for i = 1:size(featureSetSplit,1)
    invSplit(i,:) = setdiff(1:length(c2Types),featureSetSplit(i,:));
end

load([resDir 'imageNet_' prefix '_summary_struct.mat'],'data','validationData');

d{1} = add_results(data,fsTypes{1});
d{2} = add_results(data,fsTypes{2});
d{3} = add_results(validationData,fsTypes{3});
d{4} = add_results(validationData,fsTypes{4});
d{5} = add_results(validationData,fsTypes{5});
d{6} = add_results(validationData,fsTypes{6});
d{7} = add_results(validationData,fsTypes{7});

fprintf('starting to create the ld\n');
c = 0;
for iFS = 1:2
    for iCat = 1:length(c2Types)
        for iTrain = 1
            for iSplit = 1:length(splitTypes)
                c = c+1;
                ld(c).featureSet = fsTypes{iFS};
                ld(c).category = c2Types{iCat};
                ld(c).nTrainingExamples = teTypes(iTrain);
                ld(c).split = iSplit;
                ld(c).repetition = 1;
                ld(c).dprime = d{iFS}.dprimes(iCat,iTrain,iSplit);
            end
        end
    end
    fprintf('%d\n',iFS);
end

for iFS = 3:7
    for iThresh = 1:length(d{iFS}.possibilities)
        for iCat = 1:size(invSplit,2)
            for iTrain = 1
                for iSplit = 1:length(splitTypes)
                    for iRep = 1:size(invSplit,1)
                        c = c+1;
                        ld(c).featureSet = [fsTypes{iFS} '-' d{iFS}.possibilities{iThresh}];
                        ld(c).category = c2Types{invSplit(iRep,iCat)};
                        ld(c).nTrainingExamples = teTypes(iTrain);
                        ld(c).split = iSplit;
                        ld(c).repetition = iRep;
                        ld(c).dprime = d{iFS}.allData(iRep,iThresh,iCat,iTrain,iSplit);
                    end
                end
            end
        end
    end
    fprintf('%d\n',iFS);
end

results.data = d;
results.(['threshold_' prefix '_struct']) = ld;
clear d ld;

fprintf('saving the mat file\n');
save([outDir 'threshold_' prefix '_summary.mat'],'-struct','results','data');
save([outDir 'threshold_' prefix '_struct.mat'],'-struct','results',['threshold_' prefix '_struct']);
fprintf('creating the table & saving the table file\n');
t = struct2table(results.(['threshold_' prefix '_struct']));
writetable(t,[outDir 'threshold_' prefix '_table.csv']);
end

function result = add_results(d,name)
    result = [];
    for i = 1:length(d)
        if strcmp(d(i).names,name)
            result = d(i);
        end
    end
end
