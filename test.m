load('data/gradeass3.mat');
load('data/ground3.mat');
load('data/gradee3.mat');
load('data/gradee_member_grade3');
load('data/gradee_member_grader3');
load('data/grader_member_grade3');
load('data/grader_member_gradee3'); 

rank_gradee_by_grader_pairwise_3 = generateOrdinal(numgrader, gradee3, gradeass3,  grader_member_grade3);
epoch=1; 
  maxepoch= 30; % number of iterations
  
  lamda_tau =  2; % regularization parameter for L2
  disp('-----------------------------------------------');
  fprintf('When regularization lambda_tau is %d  \r', 2);
  sigma = 1;
  sigma_tau = 1;
  lamda = 2;

validation_btmg3_pg4_btmg = [];
validation_btmg3_pg5_btmg = [];

RMSE3_pg4_btmg = [];
RMSE3_pg5_btmg = [];

% number of iteration for averaging          
for itr = 1 : 10
    true_prev4_btmg = normrnd(result3_pg4, sigma); % prior, and update true score
    true_prev5_btmg = normrnd(result3_pg5, sigma); % prior, and update true score
    relia3 = normrnd(1, sigma_tau, numgrader, 1); % prior, grader reliability
    temp_pg4 = [];
    temp_pg5 = [];
    
     % Prepare the training observations in a vector 
    numOfPair3 = 0;
    observation3 = [];
    observation3_grader = [];
    for i = 1: numgrader
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

% number of epoch for converging   
for epoch = 1:maxepoch
    
        %random shuffle the observations
    rr = randperm(numOfPair3)';
    observation3 = observation3(rr,:);
    observation3_grader = observation3_grader(:,rr);
    
   eta  = 2/sqrt(epoch); % Learning rate
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%
            %%%%%%%%%%%%%%%%% BTMG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
            true_prev4_btmg = btmg_train(true_prev4_btmg, numOfPair3, observation3, observation3_grader, result3_pg4, relia3, sigma, sigma_tau, lamda, lamda_tau, eta); %%% add pg_4 as prior
            true_prev5_btmg = btmg_train(true_prev5_btmg, numOfPair3, observation3, observation3_grader, result3_pg5, relia3, sigma, sigma_tau, lamda, lamda_tau, eta); %%% add pg_5 as prior
            %%
 
           %%%%%%%%%% compute validatoin accuracy %%%%%%%
            correct_pg4_btmg = 0;
            correct_pg5_btmg = 0;
            
            PG4_btmg = [];
            PG5_btmg = [];

            for i =1: length(ground3)
                PG4_btmg = [PG4_btmg; true_prev4_btmg(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(i,1))), 3))))];     
                PG5_btmg = [PG5_btmg; true_prev5_btmg(find(ismember(gradee3, gradeass3(min(find(cell2mat(gradeass3(:,1)) == ground3(i,1))), 3))))];
            end

            result_ground3  = [ground3, PG4_btmg, PG5_btmg];
            sorted_ground3 = sortrows(result_ground3, 2);
            
            total = 0;
            for p = 1:22
                for q = p+1 : 23
                    if(sorted_ground3(q,2) > sorted_ground3(p,2))
                        total  = total + 1;
                        if(sorted_ground3(q,4) > sorted_ground3(p,4))
                            correct_pg4_btmg = correct_pg4_btmg + 1;
                        end
                        if(sorted_ground3(q,5) > sorted_ground3(p,5))
                            correct_pg5_btmg = correct_pg5_btmg + 1;
                        end
                    end
                end
            end

            temp_pg4 = [temp_pg4, correct_pg4_btmg / total];
            
            temp_pg5 = [temp_pg5, correct_pg5_btmg / total];
    
            fprintf(1,'epoch %d  PG4 %6.4f  PG5 %6.4f \r',epoch, correct_pg4_btmg / total, correct_pg5_btmg / total);

end

RMSE3_pg4_btmg = [RMSE3_pg4_btmg; sqrt(mean((result_ground3(:,4) - result_ground3(:,2)).^2))];
RMSE3_pg5_btmg = [RMSE3_pg5_btmg; sqrt(mean((result_ground3(:,5) - result_ground3(:,2)).^2))];

validation_btmg3_pg4_btmg = [validation_btmg3_pg4_btmg; temp_pg4];
validation_btmg3_pg5_btmg = [validation_btmg3_pg5_btmg; temp_pg5];

end
fprintf('BTMG+PG4 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg4_btmg));
fprintf('BTMG+PG5 cardinal evluation RMSE %6.4f \r', mean(RMSE3_pg5_btmg));
fprintf('BTMG+PG4 ordinal evluation RMSE %6.4f \r', mean(validation_btmg3_pg4_btmg(:,maxepoch)));
fprintf('BTMG+PG5 ordinal evluation RMSE %6.4f \r', mean(validation_btmg3_pg5_btmg(:,maxepoch)));