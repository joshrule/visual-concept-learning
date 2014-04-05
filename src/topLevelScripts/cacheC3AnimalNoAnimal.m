home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));
imgSetDir = [home 'imageSets/animalNoAnimal/'];
c2Dir = [imgSetDir 'c2Cache/'];
outDir = [imgSetDir 'c2Cache/'];
c2PatchSet = 'kmeans';
classes = {'animals' 'noAnimals'};
type1 = {'organic','inorganic'};
type2 = {'isolated','shared'};

for c = 1:length(classes)
    load([c2Dir classes{c} '.' c2PatchSet '.c2.mat'],'c2');
    for i = 1:length(type1)
        for j = 1:length(type2)
            fprintf('%s - %s\n',type1{i},type2{j});
            load([home 'patchSets/' type1{i} 'C3.' type2{j} '.mat'], ...
	         'models','method');
            c3 = testC3(c2,models,method);
            save([outDir classes{c} '.' type1{i} '.' type2{j} '.c3.mat'],'c3');
            clear c3;
        end
    end
end
