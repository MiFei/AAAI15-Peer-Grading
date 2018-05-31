%%% function for computing ordinal evaluation, and also return the
%%% predicted score on the ground truth set
function [ result, PG_score_ground ] = ordinal_eva(grade, gradee, ground, predicted_score )
%   result: returned ordinal evaluation;
%   PG_score_ground: predicted score on the ground truth set
%   grade: peer grading data set
%   gradee: set of gradees
%   ground: the ground truth set
%   predicted_score: predicted score by some model

correct_pg_ordinal = 0;

PG_score_ground = [];

for i =1: length(ground)
    PG_score_ground = [PG_score_ground; predicted_score(find(ismember(gradee, grade(min(find(cell2mat(grade(:,1)) == ground(i,1))), 3))))];
end

result_ground  = [ground, PG_score_ground];
sorted_ground = sortrows(result_ground, 2);

total = 0;
for p = 1:22
    for q = p+1 : 23
        if(sorted_ground(q,2) > sorted_ground(p,2))
            total  = total + 1;
            if(sorted_ground(q,4) > sorted_ground(p,4))
                correct_pg_ordinal = correct_pg_ordinal + 1;
            end
        end
    end
end

result = correct_pg_ordinal / total;
