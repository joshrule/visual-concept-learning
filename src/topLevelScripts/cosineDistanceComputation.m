function modelSimilarities = cosineDistanceComputation(y1,y2)
% y1, a matrix giving a set of N F-vectors
% y2, a matrix giving a set of M F-vectors
    norm1 = normr(y1); % NxF
    norm2 = normr(y2); % MxF
    modelSimilarities = ((norm1*norm2')+1)./2; % NxM
end

% function s = cosineSimilarity(v1,v2)
%     norm1 = sqrt(dot(v1,v1));
%     norm2 = sqrt(dot(v2,v2));
%     s = dot(v1,v2)/(norm1*norm2);
% end
% 
% function y = col(x)
%     y = reshape(x,[],1);
% end
% 
% function y = row(x)
%     y = reshape(x,1,[]);
% end
