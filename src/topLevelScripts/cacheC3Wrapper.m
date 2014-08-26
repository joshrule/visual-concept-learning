function cacheC3Wrapper(c3Files,c2Files,modelDir)
load([modelDir 'models.mat'],'models');
for i = 1:length(c3Files)
    if ~exist(c3Files{i},'file')
        cacheC3(c3Files{i},c2Files{i}, ...
          [modelDir 'models.mat'],[modelDir 'setup.mat'],models);
        fprintf('%d: cached %s\n',i,c3Files{i});
    else
        fprintf('%d: found %s\n',i,c3Files{i});
    end
end
