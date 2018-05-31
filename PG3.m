load('data/gradeass3.mat');
%%% Test PG3 individually

load('data/ground3.mat');
load('data/gradee3.mat');
load('data/grader3.mat');
load('data/gradee_member_grade3');
load('data/gradee_member_grader3');
load('data/grader_member_grade3');
load('data/grader_member_gradee3');


[bias3_pg3, result3_pg3]  = gibbs1 (11, 0.01, 0.1, 0.1, 0.1, 25, 300, gradeass3, grader3, gradee_member_grade3, gradee_member_grader3, grader_member_grade3, grader_member_gradee3);
burn_in_size3 = 0.3 * size(result3_pg3, 2);
estimate3_pg3 = mean(result3_pg3(:,(burn_in_size3+1): size(result3_pg3, 2)), 2);
result3_pg3 = mean(estimate3_pg3,2);

s = length(ground3);
pg3 = [];

for it =1: s
    pg3 = [pg3; estimate3_pg3(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(it,1))), 3))))];
end

result_ground3  = [ground3, pg3];

RMSE3_baseline = sqrt(mean((result_ground3(:,3) - result_ground3(:,2)).^2));
RMSE3_pg3 = sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2));

fprintf('PG3 cardinal evluation RMSE %6.4f \r', RMSE3_pg3);
