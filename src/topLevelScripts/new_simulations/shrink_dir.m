function shrink_dir(topdir)
    dir_data = dir(topdir);
    for i = 1:length(dir_data)
        if ~strcmp(dir_data(i).name, '.') && ~strcmp(dir_data(i).name, '..')
            name = [topdir dir_data(i).name];
            if dir_data(i).isdir()
                fprintf('dir: %s\n', name);
                shrink_dir([name '/']);
            elseif length(regexp(name, '.+\.mat$', 'match')) > 0
                fprintf('mat: %s\n', name);
                data = load(name);
                save(name, '-struct', 'data');
            end
        end
    end
end
