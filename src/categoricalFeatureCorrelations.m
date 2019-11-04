function categoricalFeatureCorrelations(labels,imgNames,c2,c3,c3Wnids, ...
  targetWnids,N,outDir,simFile,maxSize)
% categoricalFeatureCorrelations(labels,imgNames,c2,c3,c3Wnids, ...
%   targetWnids,N,outDir,simFile)
%
% compute the strength of feature correlations with a given target category
%
% labels: [nClasses nImgs] matrix, the labels according to outputWnid
% imgNames: nImgs cell vector of strings, the image filenames
% c2: [nC2Features nImgs] array, the C2 responses
% c3: [nC3Features nImgs] array, the C3 responses
% c3Wnids: nC3Features cell vector of strings, WordNet IDs for C3 categories
% targetWnids: nClasses cell vector of strings, WordNet IDs for output categories
% N: scalar, the number of categories to compare for C3 features
% outDir: string, in which directory on disk to write the results
% simFile: string, where on disk to find the perl file allowing semantic
%   distance calculations.
% maxSize: scalar, maximum edge length when resizing images
    a = tic;
    meanDistances(c2,labels,N,[outDir 'c2Corrs.mat']);
    fprintf('categorical C2 elapsed time %.3f\n',toc(a));
    b = tic;
    meanDistances(c3,labels,N,[outDir 'c3Corrs.mat'], ...
      c3Wnids,targetWnids,simFile);
    fprintf('categorical C3 elapsed time %.3f\n',toc(b));
end

function meanDistances(features,labels,N,outFile,wnids,targets,simFile)
% a helper function for the above
    corrs = corr(features',labels');
    nClasses = size(labels,1);
    bestChoices = zeros(N,nClasses);
    randChoices = zeros(N,nClasses);
    bestCorrs = zeros(N,nClasses);
    randCorrs = zeros(N,nClasses);
    bestWnids = cell(N,nClasses);
    randWnids = cell(N,nClasses);
    bestDist = zeros(N,nClasses);
    randDist = zeros(N,nClasses);
    for iClass = 1:nClasses
	fprintf('%d',iClass);
        [~,idx] = sort(squeeze(corrs(:,iClass)),'descend'); 
        bestChoices(:,iClass) = idx(1:N);
        randChoices(:,iClass) = randperm(length(idx),N);
        bestCorrs(:,iClass) = corrs(bestChoices(:,iClass),iClass);
        randCorrs(:,iClass) = corrs(randChoices(:,iClass),iClass);
        if (nargin > 4)
            bestWnids(:,iClass) = wnids(bestChoices(:,iClass));
            randWnids(:,iClass) = wnids(randChoices(:,iClass));
	    bestDist2 = zeros(1,N);
	    randDist2 = zeros(1,N);
	    a = tic;
            parfor i = 1:N
               bestDist2(i) = str2num(perl(simFile,targets{iClass},bestWnids{i,iClass}));
               randDist2(i) = str2num(perl(simFile,targets{iClass},randWnids{i,iClass}));
            end
	    fprintf(' %.3fs',toc(a));
	    bestDist(:,iClass) = bestDist2;
	    randDist(:,iClass) = randDist2;
        end
	fprintf('\n');
    end
    save(outFile,'corrs','bestChoices','randChoices','bestCorrs','randCorrs');
    if (nargin > 4)
        save(outFile,'bestWnids','randWnids','bestDist','randDist','-append');
    end
end
