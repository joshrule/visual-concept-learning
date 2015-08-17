function setupVisualSimilarities(p,basetype,c2Files)
    function [s,v] = computeVVTEntry(s,i,j)
        if isnan(s.vocabVsTestVisualSimilarities(i,j))
            iCat = ceil(i/p.nImgs);
            testC2 = buildC2(c2Files(iCat));
            startI = (iCat-1)*p.nImgs+1;
            stopI = iCat*p.nImgs;
            s.vocabVsTestVisualSimilarities(:,startI:stopI) = ...
              [corr(oC2,testC2); corr(iC2,testC2)];
        end
        v = s.vocabVsTestVisualSimilarities(i,j);
    end

    function [s,v] = computeTVTEntry(s,i,j)
        if isnan(s.testVsTestVisualSimilarities(i,j))
            iCat = ceil(i/p.nImgs);
            jCat = ceil(j/p.nImgs);
            testC2I = buildC2(c2Files(iCat));
            testC2J = buildC2(c2Files(jCat));
            startI = (iCat-1)*p.nImgs+1;
            stopI = iCat*p.nImgs;
            startJ = (jCat-1)*p.nImgs+1;
            stopJ = jCat*p.nImgs;
            tmp = corr(testC2I,testC2J);
            s.testVsTestVisualSimilarities(startI:stopI,startJ:stopJ) = tmp;
            s.testVsTestVisualSimilarities(startJ:stopJ,startI:stopI) = tmp';
        end
        v = s.testVsTestVisualSimilarities(i,j);
    end

    function [s,v] = computeVVVEntry(s,i,j)
        if isnan(s.vocabVsVocabVisualSimilarities(i,j))
            startI = (i-1)*p.nImgs+1;
            stopI = i*p.nImgs;
            startJ = (j-1)*p.nImgs+1;
            stopJ = j*p.nImgs;
            c2 = [oC2 iC2];
            testC2I = c2(:,startI:stopI);
            testC2J = c2(:,startJ:stopJ);
            tmp = mean(mean(corr(testC2I,testC2J)));
            s.vocabVsVocabVisualSimilarities(i,j) = tmp;
            s.vocabVsVocabVisualSimilarities(j,i) = tmp;
        end
        v = s.vocabVsVocabVisualSimilarities(i,j);
    end

    outFile = [p.outDir 'visual-similarities.mat'];
    if ~exist([outFile '.flag'],'file')
        organicModelData = load([p.caching.patchDir basetype '-organicC3v' p.suffix '/setup.mat'],'c2');
        inorganicModelData = load([p.caching.patchDir basetype '-inorganicC3v' p.suffix '/setup.mat'],'c2');
        oC2 = organicModelData.c2;
        iC2 = inorganicModelData.c2;
    
        fprintf('setting up visual similarities\n');

        s = getContentsOfFile(outFile);

        fprintf('check\n');

        var = 'testVsTestVisualSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(p.nImgs*length(c2Files));
            s.([var 'f']) = @computeTVTEntry;
        end
        fprintf('check\n');
        var = 'vocabVsTestVisualSimilarities';
        if ~isfield(s,var)
            s.(var) = nan(size(oC2,2)+size(iC2,2), p.nImgs*length(c2Files));
            s.([var 'f']) = @computeVVTEntry;
        end
        fprintf('check\n');

        var = 'vocabVsVocabVisualSimilarities';
        if ~isfield(s,var)
            s.(var) = nan((size(oC2,2)+size(iC2,2))/p.nImgs);
        end
        fprintf('check\n');

        var = 'vocabVsVocabVisualSimilarities';
        if ~isfield(s,[var 'f'])
            s.([var 'f']) = @computeVVVEntry;
        end
        fprintf('check\n');
        save(outFile,'-v7.3','-append','-struct','s','vocabVsVocabVisualSimilaritiesf');

        if isnan(s.testVsTestVisualSimilarities(end,end)) || ...
           isnan(s.vocabVsTestVisualSimilarities(end,end))
            fprintf('values initialized, but still need to cache computations\n');
            for i = 1:length(c2Files)
                if mod(i,10) == 0, fprintf('%d\n',i); end;
                s = computeVVTEntry(s,1,i*p.nImgs);
                for j = 1:length(c2Files)
                    s = computeTVTEntry(s,i*p.nImgs,j*p.nImgs)
                end
                save(outFile,'-v7.3','-append','-struct','s', ...
                  'testVsTestVisualSimilarities','vocabVsTestVisualSimilarities');
            end
        end
        fprintf('check\n');
        system(['touch ' outFile '.flag']);
    end
end
