% Generate oridinal preferences from cardinal peer grading scores
function [rank_gradee_by_grader_pairwise] = generateOrdinal(numgrader, gradee, grade,  grader_member_grade)
%   rank_gradee_by_grader_pairwise: returned ordinal preferences;
%   numgrader: number of grader of this assignment
%   gradee: gradee matrix
%   grade: peer grading data set
%   gradee: set of gradees
%   grader_member_grade: membership matrix
rank_gradee_by_grader_pairwise = {};
for i = 1 : numgrader
    score_given_by_i =   cell2mat(grade(grader_member_grade(:,i),4));
    gradee_graded_by_i = grade(grader_member_grade(:,i),3);
    
    % sort the score given by grader i
    [sorted, index] = sort(score_given_by_i,'descend');
  
    sorted_gradee = gradee_graded_by_i(index);    
    
    % match gradee id to index¡A and find pair
    
    numOfPairwise = 0;
    for j = 1:(size(sorted_gradee)-1)
        for k = (j+1) : size(sorted_gradee)
            
            if(sorted(j)>sorted(k))
                sorted_gradee_index  = [];
                sorted_gradee_index = [sorted_gradee_index; find(ismember(gradee, sorted_gradee(j))); find(ismember(gradee, sorted_gradee(k)))];
                numOfPairwise = numOfPairwise + 1;
                %attach
                rank_gradee_by_grader_pairwise(i,numOfPairwise) = {sorted_gradee_index}; % carefull next when grader only graded one submission
 
            end
        end
    end   
end



