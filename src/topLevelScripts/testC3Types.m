home = '/home/josh/data/ruleRiesenhuber2013/';
addpath(genpath([home 'src/']));

suffixes = {'' 'HNM', 'FullHNM', 'DoubleHNM', 'OldHalfHNM', 'OldFullHNM', 'OldDoubleHNM'};

load([home 'comparisonData.mat'],'sharedC2','sharedC3','sharedCategories', ...
     'sharedImgNames','sharedC3Type');

for iSuff = 1:length(suffixes)
    for iCat = 1:5
        
        if exist([home 'imageSets/imageNet/organicC2Cache/' sharedCategories{iCat} '.inorganic' suffixes{iSuff} '.isolated.c3.mat'],'file')
            load([home 'imageSets/imageNet/organicC2Cache/' sharedCategories{iCat} '.inorganic' suffixes{iSuff} '.isolated.c3.mat'],'c3')
	else
            load([home 'imageSets/imageNet/inorganicC2Cache/' sharedCategories{iCat} '.inorganic' suffixes{iSuff} '.isolated.c3.mat'],'c3')
        end

	oldC3 = sharedC3{iCat}(:);
	newC3 = c3(:);

        distance(iSuff,iCat) = sqrt(sum((oldC3-newC3).^2));
    end
end
