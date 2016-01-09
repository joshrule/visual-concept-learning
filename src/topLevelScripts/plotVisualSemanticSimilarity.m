% a scatter plot of C2 similarity (correlation) vs semantic similarity,
% with (non)selection plotted as the third variable (i.e. color of dot)
function plotVisualSemanticSimilarity(meanActivations,thresh,semSim,c2Corrs,nImgs) 
    choose = find(meanActivations > thresh);
    ignore = find(meanActivations <= thresh);
    
    visSim = nan(2000,1);
    for i = 1:2000
        i1 = (i-1)*nImgs+1;
        i2 = i*nImgs;
        for j = 1:100
            j1 = (j-1)*nImgs+1;
            j2 = j*nImgs;
            visSim(i,j) = mean(mean(c2Corrs(i1:i2,j1:j2)));
        end
    end

    % colored by mean activation
    %scatter(semSim,visSim,6,meanActivations);
    
    % joint over visual similarity and mean activation
    %scatter(visSim,meanActivations,4,[.6 .6 .6]);
    %grid on
    
    % joint over semantic similarity and mean activation
    %scatter(semSim,meanActivations,4,[.6 .6 .6]);
    %grid on
    
    % scatter plot colored by selection
    %scatter(semSim(ignore),visSim(ignore),2,[.6 .6 .6]);
    %scatter(semSim(choose),visSim(choose),2,'r');
    %legend({'ignored','chosen'});
    
    % conditional histograms
    rawData = semSim;
    restrictedData = semSim(choose);
    altData = visSim(choose);
    start = 0.0;
    stop = 1.0;
    step = 0.01;
    steps = start:step:stop;
    visSims = cell(length(steps)-1,1);
    fprintf('visSims size: %d %d\n',size(visSims));
    ls = nan(length(steps)-1,1);
    fprintf('ls size: %d %d\n',size(visSims));
    for i = 1:length(visSims)
        idxs = find(restrictedData >= steps(i) & restrictedData < steps(i+1));
        visSims{i} = altData(idxs);
        ls(i) = length(idxs);
    end
    maxL = max(ls);
    fprintf('maxL: %d\n',maxL);
    visSimsWithNans = nan(maxL,length(steps)-1);
    for i = 1:length(visSims)
        visSimsWithNans(1:ls(i),i) = visSims{i};
    end
    fprintf('visSimsWithNans size: %d %d\n',size(visSimsWithNans));

    ps = [0.0:.1:0.8 .9:.001:1.0];
    fprintf('ps size: %d %d\n',size(ps));

    numerator = histc(visSimsWithNans,ps); % length(ps) x size(visSimsWithNans,2)
    numerator = numerator(1:(end-1),:);
    fprintf('numerator size: %d %d\n',size(numerator))

    % data1 = histc(restrictedData,start:step:stop);
    % data1 = data1(1:(end-1));
    % fprintf('data1 size: %d %d\n',size(data1))

    denominator = histc(reshape(rawData,[],1),steps);
    denominator = repmat(denominator(1:(end-1))',length(ps)-1,1);
    fprintf('denominator size: %d %d\n',size(denominator))

    % data = data1(1:(end-1)) ./ data2(1:(end-1));
    data = numerator ./ denominator;
    fprintf('data size: %d %d\n',size(data))

    clf;
    hold on;
    % stacked bar graphs stack across rows; each column is another bar
    bar((start+step/2):step:(stop-step/2),data',1,'stacked','EdgeColor','none','LineStyle','none');
    axis([start-step stop+step 0 1]);
    subPs = [1:10:(length(ps)-1) length(ps)-1];
    idxPs = [1:10:(length(ps)-1) length(ps)];
    cbh = colorbar('YGrid','on');
    set(cbh,'ytick',subPs);
    set(cbh,'yticklabel',arrayfun(@num2str,ps(idxPs),'uni',false));
    xlabel('Semantic Similarity (Wu-Palmer Score)');
    ylabel(['p(mean activation > ' num2str(thresh) ' | semantic similarity)']);
    title(['All Categories, p(mean activation > ' num2str(thresh) ' | semantic similarity), colored by visual similarity, all data']);
    print(gcf,'-depsc','~/cond-mean-semantics.eps');
    hold off;
    clear rawData start stop step data1 data2 data

    % conditional histograms
    rawData = visSim;
    restrictedData = visSim(choose);
    altData = semSim(choose);
    start = 0.90;
    stop = 1.00;
    step = 0.001;
    steps = start:step:stop;
    visSims = cell(length(steps)-1,1);
    ls = nan(length(steps)-1,1);
    for i = 1:length(visSims)
        idxs = find(restrictedData >= steps(i) & restrictedData < steps(i+1));
        visSims{i} = altData(idxs);
        ls(i) = length(idxs);
    end
    maxL = max(ls);
    visSimsWithNans = nan(maxL,length(steps)-1);
    for i = 1:length(visSims)
        visSimsWithNans(1:ls(i),i) = visSims{i};
    end

    ps = 0.0:.01:1.0;

    numerator = histc(visSimsWithNans,ps); % length(ps) x size(visSimsWithNans,2)
    numerator = numerator(1:(end-1),:);

    % data1 = histc(restrictedData,start:step:stop);
    % data1 = data1(1:(end-1));
    % fprintf('data1 size: %d %d\n',size(data1))

    denominator = histc(reshape(rawData,[],1),steps);
    denominator = repmat(denominator(1:(end-1))',length(ps)-1,1);

    % data = data1(1:(end-1)) ./ data2(1:(end-1));
    data = numerator ./ denominator;

    clf;
    hold on;
    % stacked bar graphs stack across rows; each column is another bar
    bar((start+step/2):step:(stop-step/2),data',1,'stacked','EdgeColor','none','LineStyle','none');
    axis([start-step stop+step 0 1]);
    subPs = [1:10:(length(ps)-1) length(ps)-1];
    idxPs = [1:10:(length(ps)-1) length(ps)];
    cbh = colorbar('YGrid','on');
    set(cbh,'ytick',subPs);
    set(cbh,'yticklabel',arrayfun(@num2str,ps(idxPs),'uni',false));
    xlabel('Visual Similarity (C2 Correlation)');
    ylabel(['p(mean activation > ' num2str(thresh) ' | visual similarity)']);
    title(['All Categories, p(mean activation > ' num2str(thresh) ' | visual similarity)']);
    print(gcf,'-depsc','~/cond-mean-visual.eps');
    hold off;
    clear rawData start stop step data1 data2 data
    
    [~,p,ci] = ttest2(semSim(choose),semSim(ignore));
    fprintf('Semantic Test: p = %E, ci = [%.3f, %.3f]\n',p,ci);
    [~,p,ci] = ttest2(visSim(choose),visSim(ignore));
    fprintf('Visual Test: p = %E, ci = [%.3f, %.3f]\n',p,ci);
end
