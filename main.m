load('data/gradeass3.mat');
load('data/ground3.mat');
load('data/gradee3.mat');
load('data/grader3.mat');
load('data/gradee_member_grade3');
load('data/gradee_member_grader3');
load('data/grader_member_grade3');
load('data/grader_member_gradee3');

numgrader3 = size(grader_member_gradee3, 2);
rank_gradee_by_grader_pairwise_3 = generateOrdinal(numgrader3, gradee3, gradeass3,  grader_member_grade3);

disp('******************** Cardinal Models ************************');
[bias3_pg3, result3_pg3]  = gibbs3 (11, 0.01, 0.1, 0.1, 0.1, 25, 300, gradeass3, grader3, gradee_member_grade3, gradee_member_grader3, grader_member_grade3, grader_member_gradee3);
burn_in_size3 = 0.3 * size(result3_pg3, 2);
estimate3_pg3 = mean(result3_pg3(:,(burn_in_size3+1): size(result3_pg3, 2)), 2);
result3_pg3 = mean(estimate3_pg3,2);

PG3 = [];

for it =1: length(ground3)
    PG3 = [PG3; estimate3_pg3(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(it,1))), 3))))];
end

result_ground3  = [ground3, PG3];

RMSE3_b11aseline = sqrt(mean((result_ground3(:,3) - result_ground3(:,2)).^2));
RMSE3_pg3 = sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2));

[ ordinal_pg3, PG3 ] = ordinal_eva(gradeass3, gradee3, ground3, result3_pg3 );

fprintf('PG3 cardinal evluation RMSE: %6.4f \r', RMSE3_pg3);
fprintf('PG3 ordinal evluation: %6.4f \r', ordinal_pg3);

% for i = 6:7
% fprintf('When lambda for pg5 and beta for pg4 is %d  \r\r', 100*i);

[bias3_pg4, relia3_pg4, result3_pg4]  = gibbs4 (11, 0.04, 600, 0.04, 300, gradeass3,  gradee_member_grade3, gradee_member_grader3, grader_member_grade3, grader_member_gradee3);
burn_in_size3 = 0.3 * size(result3_pg4, 2);
estimate3_pg4 = mean(result3_pg4(:,(burn_in_size3+1): size(result3_pg4, 2)), 2);
result3_pg4 = mean(estimate3_pg4,2);

PG4 = [];

for it =1: length(ground3)
    PG4 = [PG4; estimate3_pg4(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(it,1))), 3))))];
end

result_ground3  = [ground3, PG4];

RMSE3_pg4 = sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2));

[ ordinal_pg4, PG4 ] = ordinal_eva(gradeass3, gradee3, ground3, result3_pg4 );

fprintf('PG4 cardinal evluation RMSE: %6.4f \r', mean(RMSE3_pg4));
fprintf('PG4 ordinal evluation: %6.4f \r', mean(ordinal_pg4));

[bias3_pg5, relia3_pg5, result3_pg5]  = gibbs5 (11, 0.04, 0.04, 0.1 , 700, 300, gradeass3,  gradee_member_grade3, gradee_member_grader3, grader_member_grade3, grader_member_gradee3);
burn_in_size3 = 0.3 * size(result3_pg5, 2);
estimate3_pg5 = mean(result3_pg5(:,(burn_in_size3+1): size(result3_pg5, 2)), 2);
result3_pg5 = mean(estimate3_pg5,2);

PG5 = [];

for it =1: length(ground3)
    PG5 = [PG5; estimate3_pg5(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(it,1))), 3))))];
end

result_ground3  = [ground3, PG5];

RMSE3_pg5 = sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2));

[ ordinal_pg5, PG5 ] = ordinal_eva(gradeass3, gradee3, ground3, result3_pg5 );
fprintf('PG5 cardinal evluation RMSE: %6.4f \r', mean(RMSE3_pg5));
fprintf('PG5 ordinal evluation: %6.4f \r', mean(ordinal_pg5));

save('data/result_pg3', 'result_pg3');
save('data/result_pg4', 'result_pg4');
save('data/result_pg5', 'result_pg5');
%%
disp('******************** Cardinal+Ordinal Models ************************');
disp('******************** BTM ************************');
maxepoch= 50; % number of iterations
% 
% lambda =  j*0.2; % regularization parameter for L2
% disp('-----------------------------------------------');
% fprintf('When regularization lambda is %d  \r', 0.2*j);
sigma = 1;

validation_btm3_pg3_btm = [];
validation_btm3_pg4_btm = [];
validation_btm3_pg5_btm = [];
RMSE3_pg3_btm = [];
RMSE3_pg4_btm = [];
RMSE3_pg5_btm = [];

%number of iteration for averaging
for itr = 1 : 10
    
    %     temp_pg3_btm = [];
    %     temp_pg4_btm = [];
    %     temp_pg5_btm = [];
    
    %Prepare the training observations in a vector
    numOfPair3 = 0;
    observation3 = [];
    observation3_grader = [];
    for i = 1: numgrader3
        for j = 1 : length(rank_gradee_by_grader_pairwise_3(i,:))
            if(~isempty(rank_gradee_by_grader_pairwise_3{i,j}))
                numOfPair3 = numOfPair3 + 1;
                observation3{numOfPair3} = rank_gradee_by_grader_pairwise_3{i,j};
                observation3_grader{numOfPair3} = i;
            else
                break;
            end
            
        end
    end
    observation3 = observation3';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%% BTM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    true_prev3_btm = btm_train(maxepoch, numOfPair3, observation3, result3_pg3, 0.4, sigma ); %%% add pg_4 as prior
    true_prev4_btm = btm_train(maxepoch, numOfPair3, observation3, result3_pg4, 0.4, sigma); %%% add pg_4 as prior
    true_prev5_btm = btm_train(maxepoch, numOfPair3, observation3, result3_pg5, 0.8, sigma); %%% add pg_5 as prior
    
    
    %%%%%%%%% compute validatoin accuracy %%%%%%%
    [ temp_pg3_btm, PG3_btm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev3_btm );
    [ temp_pg4_btm, PG4_btm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev4_btm );
    [ temp_pg5_btm, PG5_btm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev5_btm );
    
    result_ground3  = [ground3, PG3_btm, PG4_btm, PG5_btm];
    
    RMSE3_pg3_btm = [RMSE3_pg3_btm; sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2))];
    RMSE3_pg4_btm = [RMSE3_pg4_btm; sqrt(mean((result_ground3(:,5) - result_ground3(:,2)).^2))];
    RMSE3_pg5_btm = [RMSE3_pg5_btm; sqrt(mean((result_ground3(:,6) - result_ground3(:,2)).^2))];
    
    validation_btm3_pg3_btm = [validation_btm3_pg3_btm; temp_pg3_btm];
    validation_btm3_pg4_btm = [validation_btm3_pg4_btm; temp_pg4_btm];
    validation_btm3_pg5_btm = [validation_btm3_pg5_btm; temp_pg5_btm];
    
end
fprintf('BTM+PG3 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg3_btm));
fprintf('BTM+PG4 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg4_btm));
fprintf('BTM+PG5 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg5_btm));
fprintf('BTM+PG3 ordinal evluation: %6.4f \r', mean(validation_btm3_pg3_btm));
fprintf('BTM+PG4 ordinal evluation: %6.4f \r', mean(validation_btm3_pg4_btm));
fprintf('BTM+PG5 ordinal evluation: %6.4f \r', mean(validation_btm3_pg5_btm));
%
%%
%%%%%%%%%%%%%%%%%%%%%%%% For BTM+G %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('******************** BTM+G ************************');
maxepoch= 50; % number of iterations

%   lambda_tau =  3*0.5; % regularization parameter for L2
%   disp('-----------------------------------------------');
%   fprintf('When regularization lambda_tau is %d  \r', 0.5*3);
sigma = 1;
sigma_tau = 1;

validation_btmg3_pg3_btmg = [];
validation_btmg3_pg4_btmg = [];
validation_btmg3_pg5_btmg = [];

RMSE3_pg3_btmg = [];
RMSE3_pg4_btmg = [];
RMSE3_pg5_btmg = [];


%number of iteration for averaging
for itr = 1 : 10
    relia3 = normrnd(1, sigma_tau, numgrader3, 1); % prior, grader reliability
    %     temp_pg3_btmg = [];
    %     temp_pg4_btmg = [];
    %     temp_pg5_btmg = [];
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %%%%%%%%%%%%%%%% BTMG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    true_prev3_btmg = btmg_train(maxepoch, numOfPair3, observation3, observation3_grader, result3_pg3, relia3, sigma, sigma_tau, 2, 2); %%% add pg_4 as prior
    true_prev4_btmg = btmg_train(maxepoch, numOfPair3, observation3, observation3_grader, result3_pg4, relia3, sigma, sigma_tau, 2, 2); %%% add pg_4 as prior
    true_prev5_btmg = btmg_train(maxepoch, numOfPair3, observation3, observation3_grader, result3_pg5, relia3, sigma, sigma_tau, 2, 2); %%% add pg_5 as prior
    
    
    %%%%%%%%% compute validatoin accuracy %%%%%%%
    [ temp_pg3_btmg, PG3_btmg ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev3_btmg );
    [ temp_pg4_btmg, PG4_btmg ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev4_btmg );
    [ temp_pg5_btmg, PG5_btmg ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev5_btmg );
    
    result_ground3  = [ground3, PG3_btmg, PG4_btmg, PG5_btmg];
    
    %fprintf(1,'epoch %d  Cost %6.4f   PG4 %6.4f  PG5 %6.4f \r',epoch, costFun, valid_accu, correct_pg5 / total);
    RMSE3_pg3_btmg = [RMSE3_pg3_btmg; sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2))];
    RMSE3_pg4_btmg = [RMSE3_pg4_btmg; sqrt(mean((result_ground3(:,5) - result_ground3(:,2)).^2))];
    RMSE3_pg5_btmg = [RMSE3_pg5_btmg; sqrt(mean((result_ground3(:,6) - result_ground3(:,2)).^2))];
    
    validation_btmg3_pg3_btmg = [validation_btmg3_pg3_btmg; temp_pg3_btmg];
    validation_btmg3_pg4_btmg = [validation_btmg3_pg4_btmg; temp_pg4_btmg];
    validation_btmg3_pg5_btmg = [validation_btmg3_pg5_btmg; temp_pg5_btmg];
    
end
fprintf('BTMG+PG3 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg3_btmg));
fprintf('BTMG+PG4 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg4_btmg));
fprintf('BTMG+PG5 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg5_btmg));
fprintf('BTMG+PG3 ordinal evluation: %6.4f \r', mean(validation_btmg3_pg3_btmg));
fprintf('BTMG+PG4 ordinal evluation: %6.4f \r', mean(validation_btmg3_pg4_btmg));
fprintf('BTMG+PG5 ordinal evluation: %6.4f \r', mean(validation_btmg3_pg5_btmg));
%%
%%%%%%%%%%%%%%%%%%%%%%%%% For RBTM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for k = 1:4
disp('******************** RBTM ************************');
maxepoch= 30; % number of iterations

%   lambda_tau =  3*0.5; % regularization parameter for L2
%   disp('-----------------------------------------------');
%   fprintf('When regularization lambda_tau is %d  \r', 0.5*3);
sigma = 1;

validation_rbtm3_pg3_rbtm = [];
validation_rbtm3_pg4_rbtm = [];
validation_rbtm3_pg5_rbtm = [];

RMSE3_pg3_rbtm = [];
RMSE3_pg4_rbtm = [];
RMSE3_pg5_rbtm = [];

% number of iteration for averaging
for itr = 1 : 10
    %     temp_pg3_rbtm = [];
    %     temp_pg4_rbtm = [];
    %     temp_pg5_rbtm = [];
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    %%%%%%%%%%%%%%%%% rbtm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    true_prev3_rbtm = rbtm_train(maxepoch, numOfPair3, observation3, observation3_grader, grader_member_gradee3, result3_pg3, 0.2, 1, 200, sigma); %%% add pg_4 as prior
    true_prev4_rbtm = rbtm_train(maxepoch, numOfPair3, observation3, observation3_grader, grader_member_gradee3, result3_pg4, 0.2, 1, 200, sigma); %%% add pg_4 as prior
    true_prev5_rbtm = rbtm_train(maxepoch, numOfPair3, observation3, observation3_grader, grader_member_gradee3, result3_pg5, 0.2, 1, 200, sigma); %%% add pg_5 as prior
    
    %%%%%%%%%% compute validatoin accuracy %%%%%%%
    [ temp_pg3_rbtm, PG3_rbtm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev3_rbtm );
    [ temp_pg4_rbtm, PG4_rbtm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev4_rbtm );
    [ temp_pg5_rbtm, PG5_rbtm ] = ordinal_eva(gradeass3, gradee3, ground3, true_prev5_rbtm );
    
    result_ground3  = [ground3, PG3_rbtm, PG4_rbtm, PG5_rbtm];
    
    %fprintf(1,'epoch %d  Cost %6.4f   PG4 %6.4f  PG5 %6.4f \r',epoch, costFun, valid_accu, correct_pg5 / total);
    RMSE3_pg3_rbtm = [RMSE3_pg3_rbtm; sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2))];
    RMSE3_pg4_rbtm = [RMSE3_pg4_rbtm; sqrt(mean((result_ground3(:,5) - result_ground3(:,2)).^2))];
    RMSE3_pg5_rbtm = [RMSE3_pg5_rbtm; sqrt(mean((result_ground3(:,6) - result_ground3(:,2)).^2))];
    
    validation_rbtm3_pg3_rbtm = [validation_rbtm3_pg3_rbtm; temp_pg3_rbtm];
    validation_rbtm3_pg4_rbtm = [validation_rbtm3_pg4_rbtm; temp_pg4_rbtm];
    validation_rbtm3_pg5_rbtm = [validation_rbtm3_pg5_rbtm; temp_pg5_rbtm];
    
end
fprintf('rbtm+PG3 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg3_rbtm));
fprintf('rbtm+PG4 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg4_rbtm));
fprintf('rbtm+PG5 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg5_rbtm));
fprintf('rbtm+PG3 ordinal evluation: %6.4f \r', mean(validation_rbtm3_pg3_rbtm));
fprintf('rbtm+PG4 ordinal evluation: %6.4f \r', mean(validation_rbtm3_pg4_rbtm));
fprintf('rbtm+PG5 ordinal evluation: %6.4f \r', mean(validation_rbtm3_pg5_rbtm));
