function [ true_prev ] = btmg_train( maxepoch, numOfPair, observation, observation_grader, prior, relia, sigma, sigma_tau, lambda, lambda_tau)
%   Summary of this function goes here:
%   Train the Bradley-Terry+Grader model (http://www.cs.cornell.edu/people/tj/publications/raman_joachims_14a.pdf) with prior score specified

%   Detailed explanation goes here:
%   true_prev: predicted score vector returned
%   maxepoch:     maximum training epoches
%   numOfPair:    number of pairwise or ordinal preferences
%   observation:    observed pairwise preferences as a vector
%   observation_grader:   the grader of each pairwise preference
%   prior: the a vector to specify the prior of the ordinal model, given by
%   cardinal models in our case
%   relia: a vector to specify the reliability of each grader
%   sigma, sigma_tau, lambda, lambda_tau:  hyperparameters of BTM+G
true_prev = normrnd(prior, sigma);
for epoch = 1:maxepoch
    fprintf('.');
    %random shuffle the observations
    rr = randperm(numOfPair)';
    observation = observation(rr,:);
    observation_grader = observation_grader(:,rr);
    
    eta  = 1/sqrt(epoch); % Learning rate
    costFun = 0;
    for i = 1:numOfPair
        
        
        %%%%%%%%%%%%%%%%% BTMG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% update score  for same mean case %%%%%%%%%%%%%%%%
        m1 = prior(observation{i}(1));
        m2 = prior(observation{i}(2));
        s1_btmg = true_prev(observation{i}(1));
        s2_btmg = true_prev(observation{i}(2));
        
        tau_i = relia(observation_grader{i});
        
        hypothesis = sigmf(tau_i*(s1_btmg- s2_btmg), [1,0]);
        
        costFun = costFun + lambda_tau/(2*sigma_tau^2)*((tau_i-1)^2) + lambda/(2*sigma^2)*((s1_btmg-m1)^2 + (s2_btmg-m2)^2) - log(hypothesis);
        
        delta_taudi = -eta * (lambda_tau /(2*sigma_tau^2)* (tau_i -1) - (s1_btmg-s2_btmg)*(1 - hypothesis) );
        delta_sdj = -eta * (lambda /(2*sigma^2)* (s1_btmg -m1)/(2 * sigma^2) - tau_i*(1 - hypothesis) );
        delta_sdl = -eta * (lambda /(2*sigma^2)* (s2_btmg -m2)/(2 * sigma^2) + tau_i*(1 - hypothesis) );
        
        
        %true_prev(find(grader_member_gradee1(:,observation_grader{i}))) = true_prev(find(grader_member_gradee1(:,observation_grader{i}))) + delta_sdi;
        true_prev(observation{i}(1))  = true_prev(observation{i}(1)) + delta_sdj;
        true_prev(observation{i}(2))  = true_prev(observation{i}(2)) + delta_sdl;
        relia(observation_grader{i}) = relia(observation_grader{i}) + delta_taudi;
        
    end
    
end
fprintf('\n');
end


