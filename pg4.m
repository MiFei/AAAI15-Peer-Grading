%%% Test PG4 individually

load('data/gradeass3.mat');
load('data/ground3.mat');
load('data/gradee3.mat');
load('data/gradee_member_grade3');
load('data/gradee_member_grader3');
load('data/grader_member_grade3');
load('data/grader_member_gradee3');


[bias3_pg4, relia3_pg4, result3_pg4]  = gibbs2 (11, 0.04, 600, 0.04, 300, gradeass3,  gradee_member_grade3, gradee_member_grader3, grader_member_grade3, grader_member_gradee3);
burn_in_size3 = 0.2 * size(result3_pg4, 2);
estimate3_pg4 = mean(result3_pg4(:,(burn_in_size3+1): size(result3_pg4, 2)), 2);
result3_pg4 = mean(estimate3_pg4,2);

s = length(ground3);
PG4 = [];

for it =1: s
    PG4 = [PG4; estimate3_pg4(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(it,1))), 3))))];
end

result_ground3  = [ground3, PG4];

RMSE3_baseline = sqrt(mean((result_ground3(:,3) - result_ground3(:,2)).^2));
RMSE3_pg4 = sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2));

fprintf('PG4 cardinal evluation RMSE %6.4f \r', RMSE3_pg4);
