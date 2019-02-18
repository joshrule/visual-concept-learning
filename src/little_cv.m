function split = little_cv(y,nPos)
    split = true(numel(y),1);
    pos = find(y);
    ignored = pos(randsample(numel(pos),numel(pos)-nPos));
    split(ignored) = false;
end
