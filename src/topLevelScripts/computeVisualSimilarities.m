function setupVisualSimilarities(p,basetype,c2Files)
    mf = matfile([p.outDir 'visual-similarities.mat'],'Writable',true);
    organicModelData = load([p.caching.patchDir basetype '-organicC3v' p.suffix '/setup.mat'],'c2');
    inorganicModelData = load([p.caching.patchDir basetype '-inorganicC3v' p.suffix '/setup.mat'],'c2');
    oC2 = organicModelData.c2;
    iC2 = inorganicModelData.c2;
    
    function v = computeVVTEntry(mf,i,j)
        if isnan(mf.vocabVsTestVisualSimilarities(i,j))
            jCat = ceil(i/p.nImgs);
            testC2 = buildC2(c2Files(jCat));
            startJ = (jCat-1)*p.nImgs+1;
            stopJ = jCat*p.nImgs;
            mf.vocabVsTestVisualSimilarities(:,startJ:stopJ) = ...
              [corr(oC2,testC2J); corr(iC2,testC2J)];
        end
        v = mf.vocabVsTestVisualSimilarities(i,j);
    end

    function v = computeTVTEntry(mf,i,j)
        if isnan(mf.testVsTestVisualSimilarities(i,j))
            iCat = ceil(i/p.nImgs);
            jCat = ceil(j/p.nImgs);
            testC2I = buildC2(c2Files(iCat));
            testC2J = buildC2(c2Files(jCat));
            startI = (iCat-1)*p.nImgs+1;
            stopI = iCat*p.nImgs;
            startJ = (jCat-1)*p.nImgs+1;
            stopJ = jCat*p.nImgs;
            tmp = corr(testC2I,testC2J);
            mf.testVsTestVisualSimilarities(startI:stopI,startJ:stopJ) = tmp;
            mf.testVsTestVisualSimilarities(startJ:stopJ,startI:stopI) = tmp';
        end
        v = mf.testVsTestVisualSimilarities(i,j);
    end

    initializeMFValue(mf,'testVsTestVisualSimilarities',nan(p.nImgs*length(c2Files)));
    initializeMFValue(mf,'vocabVsTestVisualSimilarities',nan(size(oC2,2)+size(iC2,2), p.nImgs*length(c2Files)));
    initializeMFValue(mf,'testVsTestVisualSimilaritiesf',@computeTVTEntry);
    initializeMFValue(mf,'vocabVsTestVisualSimilaritiesf',@computeVVTEntry);

    for i = 1:length(c2Files)
        if mod(i,10) == 0, fprintf('%d\n',i); end;
        computeVVTEntry(mf,1,i*p.nImgs);
        for j = 1:length(c2Files)
            computeTVTEntry(mf,i*p.nImgs,j*p.nImgs)
        end
    end
end
