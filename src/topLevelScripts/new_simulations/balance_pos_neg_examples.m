function [choices,w,all_choices] = balance_pos_neg_examples(y,n)
% given a set of examples, their labels, and the number of examples to choose 
% from each class, randomly sample that many and provide their counts.
    classes = unique(y);
    all_choices = [];
    choices = [];
    w = [];
    for iClass = 1:length(classes)
        % get the examples that belong to the class
        class_members = find(y == classes(iClass));
        % sample n of them
        chosen = randi(length(class_members),n,1);
        uniques = unique(chosen);
        % record your choices
        all_choices = [all_choices; chosen];
        choices = [choices; uniques];
        w = [w; histc(chosen,uniques)];
    end
end
