function [ true_prev ] = btm_train(maxepoch, numOfPair, observation, prior, lamda, sigma )
%   Summary of this function goes here:
%   Train the Bradley-Terry model with prior score specified

%   Detailed explanation goes here:
%   true_prev: predicted score vector returned
%   maxepoch:     maximum training epoches
%   numOfPair:    number of pairwise or ordinal preferences
%   observation:    observed pairwise preferences as a vector
%   prior: the a vector to specify the prior of the ordinal model, given by
%   cardinal models in our case
%   sigma, lambda:  hyperparameters of BTM+G
true_prev = normrnd(prior, sigma);
for epoch = 1:maxepoch
    fprintf('.');
    %random shuffle the observations
    rr = randperm(numOfPair)';
    observation = observation(rr,:);
    
    eta  = 1/sqrt(epoch); % Learning rate
    costFun = 0;
    for i = 1:numOfPair
        %%%%%%%%%%%%%%%%% BTM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% compute cost function %%%%%%%%%%%%%%%%%%%%%
        m1 = prior(observation{i}(1));
        m2 = prior(observation{i}(2));
        s1 = true_prev(observation{i}(1));
        s2 = true_prev(observation{i}(2));
        
        hypothesis = sigmf((s1- s2), [1,0]);
        
        costFun = costFun + lamda/(2*sigma^2)*((s1-m1)^2 + (s2-m2)^2) - log(hypothesis);
        
        %%%%%% compute gradient %%%%%%%%%%%%%%%
        delta_sdi = -eta * (lamda * (s1 -m1)/(2 * sigma^2) - (1 - hypothesis) );
        delta_sdj = -eta * (lamda * (s2 -m2)/(2 * sigma^2) + (1 - hypothesis) );
        
        %%%%%% update score %%%%%%%%%%%%%%%%
        true_prev(observation{i}(1))  = true_prev(observation{i}(1)) + delta_sdi;
        true_prev(observation{i}(2))  = true_prev(observation{i}(2)) + delta_sdj;
        
    end
    
end
fprintf('\n');
end


