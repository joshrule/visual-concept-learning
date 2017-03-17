function [choices,wo,xo,yo] = balance_pos_neg_examples(x,y,n)
% given a set of examples, their labels, and the number of examples to choose 
% from each class, randomly sample that many and provide their counts.
    classes = unique(y);
    choices = [];
    wo = [];
    xo = [];
    yo = [];
    for iClass = 1:length(unique(y))
        % get the examples that belong to the class
        class_members = find(y == classes(iClass));
        % sample n of them
        chosen = randi(length(class_members),n,1);
        unique_choices = class_members(unique(chosen));
        % record your choices
        choices = [choices; chosen];
        xo = [xo; x(unique_choices,:)];
        yo = [yo; reshape(y(unique_choices),[],1)];
        wo = [wo; histc(chosen,unique(chosen))];
    end
end
