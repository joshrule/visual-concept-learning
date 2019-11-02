function shrinkDir(topdir)
% shrinkDir(topdir)
%
% Shrink a directory by saving its *.mat files without the 7.3 format overhead.
%
% topdir: string, the directory to shrink (sub-directories are also shrunk).
    dir_data = dir(topdir);
    for i = 1:length(dir_data)
        if ~strcmp(dir_data(i).name, '.') && ~strcmp(dir_data(i).name, '..')
            name = [topdir dir_data(i).name];
            if dir_data(i).isdir()
                fprintf('dir: %s\n', name);
                shrinkDir([name '/']);
            elseif length(regexp(name, '.+\.mat$', 'match')) > 0
                fprintf('mat: %s\n', name);
                data = load(name);
                save(name, '-struct', 'data');
            end
        end
    end
end
