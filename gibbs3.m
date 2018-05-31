function [bias, true ] = gibbs3(mu, theta1, theta0, gamma, eta,  maxinum, T, grade, grader, gradee_member_grade, gradee_member_grader, grader_member_grade, grader_member_gradee)
%   Summary of this function goes here:
%   Perform Gibbs Sampling for model PG_3

%   Detailed explanation goes here:
%   mu:     inilialization value for hyperparameter \mu in PG_3
%   theta1:    inilialization value for hyperparameter \theta_1 in PG_3
%   theta0:    inilialization value for hyperparameter \theta_1 in PG_3
%   gamma:   inilialization value for hyperparameter \gamma in PG_3
%   eta:    inilialization value for hyperparameter \eta in PG_3
%   maximum: the maximum score of this assignment
%   T:  number of iterations
%   grade:  peer grading matrix, this is what we observed
%   grader: grader vector

%%%%% The purpose of using membership matrics are for speed-up, rather
%%%%% checking string ID among graders and gradees
%   gradee_member_grade: matrix to indicate gradee's membership in grade
%   gradee_member_grader:   matrix to indicate gradee relationship with
%   grader ( = transpose of grader_member_gradee )
%   grader_member_grade: matrix to indicate grader's membership in grade

% initialization for model parameters
disp('Train PG3');
theta_1 = theta1;
theta_0 = theta0;

numgradee = size(gradee_member_grade,2);
numgrader = size(grader_member_grade,2);

%   initialization for latent variables
s_u = randn(numgradee, 1) * sqrt(1/gamma) + mu;
b_v = randn(numgrader, 1) * sqrt(1/eta);

%   iteration for gibbs sampling
true = [];
bias = [];

%%% Before each iteration, compute member matrix

for i = 1:T
    fprintf('.');
    % for each student submission grading
    for j  = 1:numgradee
        
        temp = grade(gradee_member_grade(:,j),:);
        grade_index = find(gradee_member_grade(:,j) == 1);
        sum_R = 0;
        sum_y_first = 0;
        sum_y_last = 0;
        k_u = 0;
        
        %compute discretied approximation of posterior
        range = [1:maxinum];
        grader_index = gradee_member_grader(:,j);
        
        for p = 1 : size(temp, 1)
            sum_R = sum_R + s_u(j) * theta_1 + theta_0;
            s_v = s_u(grader_member_gradee(:,grader_index));
            if(~isempty(s_v))
                sum_y_first = sum_y_first + (s_v * theta_1 + theta_0) * (temp{p, 4} - b_v(grader_member_grade(grade_index(p), :)'));
            else
                sum_y_first = sum_y_first + (min(s_u) * theta_1 + theta_0) * (temp{p, 4} - b_v(grader_member_grade(grade_index(p), :)'));
            end
        end
        
        if(isempty(grader(gradee_member_grader(:,j)))) % this gradee didn't act as a grader
            sum_y_last = 0;
        else
            
            temp2 = grade(grader_member_grade(:,grader_index) ,:);
            %disp(temp2)
            grade_index = find(grader_member_grade(:,grader_index) == 1);
            k_u = size(temp2, 1); % number of submissions this grader graded
            
            for p = 1 : k_u
                sum_y_last = sum_y_last + theta_1 * (temp2{p,4} - (b_v(grader_index,1) + s_u(gradee_member_grade(grade_index(p), :)')))^2;
            end
        end;
        
        y = gamma * mu + sum_y_first + sum_y_last;
        R = gamma + sum_R;
        
        pdf = (theta_1 .* range + theta_0).^ (k_u/2) .* exp(-0.5 .* R .* (range - y/R).^2);
        
        cdf = cumsum(pdf);
        %disp(cdf)
        cutoff = rand(1) * max(cdf);
        %disp(cutoff)
        %disp(j);
        %disp(tau_u);
        while(isempty(find(cdf >= cutoff ,1)))
            cutoff = rand(1) * max(cdf);
        end
        
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
        
        sum_all_bias = 0;
        
        for p = 1 : nvi
            
            sum_all_bias = sum_all_bias + (s_u(gradee_member_grade(grade_index(p), :)') * theta_1 + theta_0) * (temp2{p,4} - s_u(gradee_member_grade(grade_index(p), :)'));
        end
        
        s_v = s_u(grader_member_gradee(:,k));
        if(~isempty(s_v))
            b_v(k,1) = sum_all_bias / (eta + nvi * (s_v * theta_1 + theta_0)) + randn()*sqrt(eta + nvi * (s_v * theta_1 + theta_0));
        else
            b_v(k,1) = sum_all_bias / (eta + nvi * (min(s_u) * theta_1 + theta_0)) + randn()*sqrt(eta + nvi * (min(s_u) * theta_1 + theta_0));
        end
        %tau_v(k,1) = gamrnd(alpha + 0.5 * nvi, (beta + 0.5*sum_all_reliability)^(-1));
        
    end;
    
    bias = [bias, b_v];
end
fprintf('\n');

end

