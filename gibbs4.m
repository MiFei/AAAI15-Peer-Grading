function [bias, relia, true] = gibbs4(mu, gam, beta, eta, T, grade, gradee_member_grade, gradee_member_grader, grader_member_grade, grader_member_gradee)
%   Summary of this function goes here:
%   Perform Gibbs Sampling for model PG_4

%   Detailed explanation goes here:
%   Detailed explanation goes here
%   mu:     inilialization value for hyperparameter \mu in PG_5
%   gam:    inilialization value for hyperparameter \gammma in PG_5
%   beta:   inilialization value for hyperparameter \beta in PG_5
%   eta:    inilialization value for hyperparameter \eta in PG_5
%   T:  number of iterations
%   grade:  peer grading matrix, this is what we observed

%%%%% The purpose of using membership matrics are for speed-up, rather
%%%%% checking string ID among graders and gradees
%   gradee_member_grade: matrix to indicate gradee's membership in grade
%   gradee_member_grader: matrix to indicate gradee relationship with
%   grader ( = transpose of grader_member_gradee )
%   grader_member_grade: matrix to indicate grader's membership in grade

disp('Train PG4');
numgradee = size(gradee_member_grade,2);
numgrader = size(grader_member_grade,2);

%   initialization for latent variables
s_u = randn(numgradee, 1) * sqrt(1/gam) + mu;
while(sum(s_u < 0)>0);
    s_u = randn(numgradee, 1) * sqrt(1/gam) + mu;
end
%s_u(s_u<=0) = min(s_u(s_u>0)); %% ensure that the initial score be positive
for k = 1:numgrader
    s_v = s_u(grader_member_gradee(:,k));
    if(isempty(s_v))
        tau_v(k,1) = gamrnd(min(s_u), beta^(-1));
        while(tau_v(k,1)<0)
            tau_v(k,1) = gamrnd(min(s_u), beta^(-1));
        end
        
    else
        tau_v(k,1)= gamrnd(s_v, beta^(-1));
        while(tau_v(k,1)<0)
            tau_v(k,1)= gamrnd(s_v, beta^(-1));
        end
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
        range = [1:0.1:25];
        
        for p = 1 : size(temp, 1)
            sum_tau = sum_tau + tau_v(grader_member_grade(grade_index(p), :)');
            sum_all = sum_all + tau_v(grader_member_grade(grade_index(p), :)') * (temp{p, 4} - b_v(grader_member_grade(grade_index(p), :)'));
        end
        
        y = gam * mu + sum_all;
        R = gam + sum_tau;
        
        if(isempty(tau_v(gradee_member_grader(:,j))))
            tau_u  = min(tau_v(tau_v>0));
        else
            tau_u = tau_v(gradee_member_grader(:,j));
            if(tau_u == 0)
                tau_u  = min(tau_v(tau_v>0));
            end;
        end;
        
        gamma_fun = [];
        for p =1:size(range,2)
            gamma_fun = [gamma_fun, gamma(range(p))];
        end
        
        
        pdf = (beta .^ (range)) ./ (gamma_fun) .* (tau_u .^ (range - 1)) .* normpdf(range, y/R, sqrt(1/R));
        
        
        cdf = cumsum(pdf);
        cutoff = rand(1) * max(cdf);
        %disp(R)
        
        s_u(j,1) = range(find(cdf >= cutoff ,1));
        
    end
    %disp(s_u);
    true = [true, s_u];
    
    % for each grader reliability and bias
    for k = 1:numgrader
        temp2 = grade(grader_member_grade(:,k) ,:);
        grade_index = find(grader_member_grade(:,k) == 1);
        nvi = size(temp2, 1); % number of submissions this grader graded
        %disp(temp2)
        %disp(nvi)
        sum_all_reliability = 0;
        sum_all_bias = 0;
        
        for p = 1 : nvi
            sum_all_reliability = sum_all_reliability + (temp2{p,4} - (b_v(k,1) + s_u(gradee_member_grade(grade_index(p), :)')))^2;
            sum_all_bias = sum_all_bias + tau_v(k,1) * (temp2{p,4} - s_u(gradee_member_grade(grade_index(p), :)'));
        end
        
        b_v(k,1) = sum_all_bias / (eta + nvi*tau_v(k,1)) + randn()*sqrt(1/(eta + nvi*tau_v(k,1)));
        %tau_v(k,1) = gamrnd(alpha + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
        
        s_v = s_u(grader_member_gradee(:,k));
        if(isempty(s_v))
            tau_v(k,1) = gamrnd(min(s_u) + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
            while(tau_v(k,1) <= 0)
                tau_v(k,1) = gamrnd(min(s_u) + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
            end
        else
            tau_v(k,1) = gamrnd(s_v + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
            while(tau_v(k,1) <= 0)
                tau_v(k,1) = gamrnd(s_v + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
            end
        end;
    end;
    %tau_v = (tau_v - min(tau_v))/ (max(tau_v) - min(tau_v)) + 1;
    relia = [relia, tau_v];
    bias = [bias, b_v];
    
end
fprintf('\n');

end

