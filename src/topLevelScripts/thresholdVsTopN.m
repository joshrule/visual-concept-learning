theDir = '~/data/ruleRiesenhuber2013/evaluation/5050vLinear/';
load([theDir 'splits.mat'],'cvsplit');
load([theDir 'combined-128-evaluation.mat'],'m','dprimes','labels');

[nClasses,nFeats,nSplits] = size(cvsplit);

for iClass = 1:nClasses
    for iSplit = 1:nSplits
        newM = m(:,cvsplit{iClass,1,iSplit});
        % 1=train
        y = labels(iClass,cvsplit{iClass,1,iSplit});
        [vals,indices] = sort(mean(newM(:,find(y)),2),'descend');
        chosen(iClass,1,iSplit,:) = indices(1:128);
        valsFlat(nSplits*(iClass-1)+iSplit,:) = vals(1:128);
        dprimesFlat(nSplits*(iClass-1)+iSplit) = dprimes(iClass,1,iSplit);
        clear y indices vals;
    end
end

[ds,idd] = sort(dprimesFlat,'ascend');


clear theDir iClass iSplit nFeats
