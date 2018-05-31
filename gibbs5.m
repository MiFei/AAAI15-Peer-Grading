%   Perform Gibbs Sampling for model PG_5

function [bias, relia, true] = gibbs5(mu, gamma, beta, eta, lambda, T, grade, gradee_member_grade, gradee_member_grader, grader_member_grade, grader_member_gradee)
%UNTITLED Summary of this function goes here
%
%   Detailed explanation goes here
%   mu:     inilialization value for hyperparameter \mu in PG_5
%   gam:    inilialization value for hyperparameter \gammma in PG_5
%   beta:   inilialization value for hyperparameter \beta in PG_5
%   eta:    inilialization value for hyperparameter \eta in PG_5
%   lambda:    inilialization value for hyperparameter \lambda in PG_5
%   T:  number of iterations
%   grade:  peer grading matrix, this is what we observed

%%%%% The purpose of using membership matrics are for speed-up, rather
%%%%% checking string ID among graders and gradees
%   gradee_member_grade: matrix to indicate gradee's membership in grade
%   gradee_member_grader: matrix to indicate gradee relationship with
%   grader ( = transpose of grader_member_gradee )
%   grader_member_grade: matrix to indicate grader's membership in grade
disp('Train PG5');
numgradee = size(gradee_member_grade,2);
numgrader = size(grader_member_grade,2);

%   initialization for latent variables
s_u = randn(numgradee, 1) * sqrt(1/gamma) + mu;
for k = 1:numgrader
    s_v = s_u(grader_member_gradee(:,k));
    if(isempty(s_v))
        tau_v(k,1) = normrnd(min(s_u), sqrt(1/beta));
        while(tau_v(k,1) < 0)
            tau_v(k,1) = normrnd(min(s_u), sqrt(1/beta));
        end
    else
        tau_v(k,1)= normrnd(s_v, sqrt(1/beta));
        while(tau_v(k,1) < 0)
            tau_v(k,1) = normrnd(s_v, sqrt(1/beta));
        end
        %disp(tau_v(k,1));
    end
end;
%tau_v = (tau_v - min(tau_v))/ (max(tau_v) - min(tau_v)) + 1;
b_v = randn(numgrader, 1) * sqrt(1/eta);




%   iteration for gibbs sampling
true = [];
relia = [];
bias = [];

%%% Before each iteration, compute member matrix

for i = 1:T
    fprintf('.');
    % for each student submission grading
    %disp(i);
    for j  = 1:numgradee
        
        temp = grade(gradee_member_grade(:,j),:);
        grade_index = find(gradee_member_grade(:,j) == 1);
        sum_tau = 0;
        sum_all = 0;
        
        %compute discretied approximation of posterior
        %range = [1:0.2:maxnum];
        
        for p = 1 : size(temp, 1)
            sum_tau = sum_tau + tau_v(grader_member_grade(grade_index(p), :)')/lambda;
            sum_all = sum_all + tau_v(grader_member_grade(grade_index(p), :)')/lambda * (temp{p, 4} - b_v(grader_member_grade(grade_index(p), :)'));
        end
        
        if(isempty(tau_v(gradee_member_grader(:,j))))
            tau_u  = min(tau_v(tau_v>0));
        else
            tau_u = tau_v(gradee_member_grader(:,j));
            if(tau_u == 0)
                tau_u  = min(tau_v(tau_v>0));
            end;
        end;
        
        y = gamma * mu + beta * tau_u + sum_all;
        R = gamma + beta + sum_tau;
        
        s_u(j,1) = normrnd(y/R, sqrt(1/R));
        if(R<0)
            disp(j)
            disp(1/R)
        end
    end
    %disp(s_u);
    true = [true, s_u];
    
    % for each grader reliability and bias
    for k = 1:numgrader
        temp2 = grade(grader_member_grade(:,k) ,:);
        grade_index = find(grader_member_grade(:,k) == 1);
        nvi = size(temp2, 1); % number of submissions this grader graded
        
        sum_all_reliability = 0;
        sum_all_bias = 0;
        
        for p = 1 : nvi
            sum_all_reliability = sum_all_reliability + (temp2{p,4} - (b_v(k,1) + s_u(gradee_member_grade(grade_index(p), :)')))^2/(2*lambda);
            sum_all_bias = sum_all_bias + tau_v(k,1)/lambda * (temp2{p,4} - s_u(gradee_member_grade(grade_index(p), :)'));
        end
        
        b_v(k,1) = normrnd(sum_all_bias/(eta + nvi*tau_v(k,1)/lambda), sqrt(1/(eta + nvi*tau_v(k,1)/lambda)));
        s_v = s_u(grader_member_gradee(:,k));
        
        %%%% discrete approximation
        tau_range = [5:0.2:60];
        
        if(isempty(s_v))
            pdf = normpdf(tau_range, (beta*min(s_u) - sum_all_reliability)/beta, sqrt(1/beta)) .* (tau_range/lambda).^(nvi/2);
        else
            pdf = normpdf(tau_range, (beta*s_v - sum_all_reliability)/beta, sqrt(1/beta)) .* (tau_range/lambda).^(nvi/2);
            %disp(tau_v(k,1));
        end;
        
        cdf = cumsum(pdf);
        cutoff = rand(1) * max(cdf);
        while(isempty(find(cdf >= cutoff ,1)))
            cutoff = rand(1) * max(cdf);
        end
        tau_v(k,1) = tau_range(find(cdf >= cutoff ,1));
    end;
    %tau_v = (tau_v - min(tau_v))/ (max(tau_v) - min(tau_v)) + 1;
    relia = [relia, tau_v];
    bias = [bias, b_v];
    
end
fprintf('\n');

end

