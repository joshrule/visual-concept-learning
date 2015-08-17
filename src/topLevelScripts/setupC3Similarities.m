function setupC3Similarities(p,organicC3Files,inorganicC3Files)
    function [s,v] = computeTVTEntry(s,i,j)
        catI = ceil(i/p.nImgs);
        catJ = ceil(j/p.nImgs);
        [c3Io,~,~] = buildC3(organicC3Files(catI));
        [c3Ii,~,~] = buildC3(inorganicC3Files(catI));
        c3I = [c3Io; c3Ii];
        [c3Jo,~,~] = buildC3(organicC3Files(catJ));
        [c3Ji,~,~] = buildC3(inorganicC3Files(catJ));
        c3J = [c3Jo; c3Ji];

        startI = (catI-1)*p.nImgs+1;
        stopI  = catI*p.nImgs;
        startJ = (catJ-1)*p.nImgs+1;
        stopJ  = catJ*p.nImgs;
        newVals = corr(c3I,c3J);
        s.testVsTestC3Similarity(startI:stopI,startJ:stopJ) = newVals;
        s.testVsTestC3Similarity(startJ:stopJ,startI:stopI) = newVals';
        v = s.testVsTestC3Similarity(i,j);
    end
    
    outFile = [p.outDir 'c3-similarities.mat'];
    if ~exist([outFile '.flag'],'file')
        fprintf('setting up C3 similarities\n');

        s = getContentsOfFile(outFile);
        
        fprintf('check\n');

        var = 'testVsTestC3Similarity';
        if ~isfield(s,var)
            s.(var) = nan(p.nImgs*length(organicC3Files));
            s.([var 'f']) = @computeTVTEntry;
            save(outFile,'-v7.3','-append','-struct','s');
        end
        fprintf('check\n');
        system(['touch ' outFile '.flag']);
    end
end
