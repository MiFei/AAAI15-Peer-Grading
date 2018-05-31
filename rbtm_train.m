function [ true_prev ] = rbtm_train(maxepoch, numOfPair, observation, observation_grader, grader_member_gradee, prior, a, lambda, lambda_a, sigma )
%   Summary of this function goes here:
%   Train the refferd Bradley-Terry model with prior score specified

%   Detailed explanation goes here:
%   true_prev: predicted score vector returned
%   maxepoch:     maximum training epoches
%   numOfPair:    number of pairwise or ordinal preferences
%   observation:    observed pairwise preferences as a vector
%   observation_grader:   the grader of each pairwise preference
%   grader_member_gradee:    membership matrix
%   prior: the a vector to specify the prior of the ordinal model, given by
%   cardinal models in our case
%   a, lambda, lambda_a, sigma:  hyperparameters of RBTM

true_prev = normrnd(prior, sigma);
for epoch = 1:maxepoch
    fprintf('.');
    costFun = 0;
    b = 0.2;
    eta  = 2/sqrt(epoch); % Learning rate
    
    %random shuffle the observations
    rr = randperm(numOfPair)';
    observation = observation(rr,:);
    observation_grader_shuffle = observation_grader(:,rr);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i = 1:numOfPair
        
        %%%%%%% pg5 %%%%%%%%
        m1 = prior(observation{i}(1));
        m2 = prior(observation{i}(2));
        s1_rbtm = true_prev(observation{i}(1));
        s2_rbtm = true_prev(observation{i}(2));
        
        if(isempty(true_prev(grader_member_gradee(:,observation_grader_shuffle{i}))))
            s_i = min(true_prev);
        else
            s_i = true_prev(grader_member_gradee(:,observation_grader_shuffle{i}));
        end
        
        hypothesis = sigmf((a * s_i + b)*(s1_rbtm- s2_rbtm), [1,0]);
        
        costFun = costFun + lambda_a* a^2 + lambda/(2*sigma^2)*((s1_rbtm-m1)^2 + (s2_rbtm-m2)^2) - log(hypothesis);
        
        s = prior(find(grader_member_gradee(:,observation_grader_shuffle{i})));
        delta_sdi = -eta * (lambda * (s_i - s)/(2 * sigma^2) - a*(s1_rbtm-s2_rbtm)*(1 - hypothesis) );
        delta_sdj = -eta * (lambda * (s1_rbtm -m1)/(2 * sigma^2) - (a * s_i + b)*(1 - hypothesis) );
        delta_sdl = -eta * (lambda * (s2_rbtm -m2)/(2 * sigma^2) + (a * s_i + b)*(1 - hypothesis) );
        true_prev(observation{i}(1))  = true_prev(observation{i}(1)) + delta_sdj;
        true_prev(observation{i}(2))  = true_prev(observation{i}(2)) + delta_sdl;
        true_prev(find(grader_member_gradee(:,observation_grader_shuffle{i}))) = true_prev(find(grader_member_gradee(:,observation_grader_shuffle{i}))) + delta_sdi;
        
    end
    
    a = 0.31;
    min_cost = Inf;
    while(a > 0)
        costFun = 0;
        
        for i = 1: ceil(numOfPair)
            
            
            s1_rbtm = true_prev(observation{i}(1));
            s2_rbtm = true_prev(observation{i}(2));
            if(isempty(true_prev(grader_member_gradee(:,observation_grader_shuffle{i}))))
                s_i = min(true_prev);
            else
                s_i = true_prev(grader_member_gradee(:,observation_grader_shuffle{i}));
            end
            
            hypothesis = sigmf((a * s_i + b)*(s1_rbtm- s2_rbtm), [1,0]);
            
            costFun = costFun  - log(hypothesis);
        end
        costFun = costFun + lambda_a* a^2 + lambda/(2*sigma^2)*sum((true_prev - prior).^2);
        if(costFun < min_cost)
            min_cost = costFun;
            a_true = a;
        end
        a = a-0.03;
    end
    
    a = a_true;
    
end
fprintf('\n');
end