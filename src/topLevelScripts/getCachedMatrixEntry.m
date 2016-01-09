function [v,val] = getCachedMatrixEntry(val,fun,var,i,j)
    s.(var) = val;
    for iI = 1:length(i)
        for iJ = 1:length(j)
            if isnan(s.(var)(i,j))
                [s,v] = fun(s,i,j);
            end
        end
    end
    val = s.(var);
    v = val(i,j);
end
