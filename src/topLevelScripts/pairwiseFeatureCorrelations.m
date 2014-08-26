function pairwiseFeatureCorrelations(imgList,c2,c3,nPairs,maxSize,simFile,outFile)
% pairwiseFeatureCorrelations(imgList,c2,c3,nPairs,maxSize,simFile,outFile)
%
% compare pairs of images using several feature sets
%
% imgList: nImgs cell vector of strings, the image filenames
% c2: [nC2Features nImgs] array, the C2 responses
% c3: [nC3Features nImgs] array, the C3 responses
% nPairs: scalar, the number of pairs to choose and compare
% maxSize: scalar, maximum edge length when resizing images
% simFile: string, where on disk to find the perl file allowing semantic
%   distance calculations.
% outFile: string, where on disk to write the results
    rngState = rng;
    pairs = {};
    for iPair = 1:nPairs
        if mod(iPair,100) == 0, fprintf('%d\n',iPair); end;
        [pairs{iPair,1},pairs{iPair,2},img1,img2] = newPair(imgList,pairs);
        c2Corr(iPair) = corr(c2(:,img1),c2(:,img2));
        c3Corr(iPair) = corr(c3(:,img1),c3(:,img2));

        wnid1 = regexp(pairs{iPair,1},'n\d+','match','once');
        wnid2 = regexp(pairs{iPair,2},'n\d+','match','once');
        semSim(iPair) = str2num(perl(simFile,wnid1,wnid2));
    end
    c2Score = corr(c2Corr,semSim);
    c3Score = corr(c3Corr,semSim);
    save(outFile,'rngState','c2Corr','c3Corr','c2Score', ...
      'c3Score','semSim','pairs');
end

function [img1,img2,ind1,ind2] = newPair(imgList,pairs)
% choose an as-yet nonexistent pair of images
%
% imgList: see above
% pairs: [nPairs 2] cell array of strings, the pair image names
%
% img1, img2: strings, the chosen images' filenames
% ind1, ind2: scalars, the chosen images' indices in 'imgList'
    stillChoosing = 1;
    while stillChoosing
        ind1 = randi(length(imgList));
        ind2 = randi(length(imgList));
        img1 = imgList{ind1};
        img2 = imgList{ind2};
        if ~strcmp(img1,img2)
            for i = 1:size(pairs,1)
                isDuplicate(i) = isequal({img1,img2},pairs(i,:)) || ...
                                 isequal({img2,img1},pairs(i,:));
            end
            if size(pairs,1) == 0 || sum(isDuplicate) == 0
                stillChoosing = 0;
            end
        end
    end
end
