Documentation, Version 1.0, 25/06/2015
-------------

Contact info: 
email: fei.mi@epfl.ch
website: https://lia.epfl.ch/~mi/

----------------------------

This version of Matlab code contains code for our AAAI 15 paper of "Probabilistic Graphical Models for Boosting
Cardinal and Ordinal Peer Grading in MOOCs" at https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9534/9280
It runs our proposed models on one of our peer grading dataset, i.e., the third assignment.

-------------
'data' folder
-------------
1. gradeass3: This matrix contains the peer grading data for this assignment.
	semantics of different columns:
	----------------------------------------------------------------------------------
	global id of the gradee | id of the grader | id of the gradee | peer grading score
	----------------------------------------------------------------------------------

2. ground3: This matrix contains the ground truth scores for this assignment.
	semantics of different columns:
	------------------------------------------------------------------------
	global id of the gradee | staff grade | taking the median of peer grades
	------------------------------------------------------------------------	

3. gradee_member_grade3, gradee_member_gradr3, grader_member_grade3, grader_member_gradee3: 
	gradee_member_grade: Matrix to indicate gradee's membership in grade
    gradee_member_grader:   Matrix to indicate gradee membership relationship with grader ( = transpose of grader_member_gradee )
    grader_member_grade: Matrix to indicate grader's membership in grade

---------
functions
---------
	gibbs3.m: Train PG3
	gibbs4.m: Train PG4
	gibbs5.m: Train PG5
	btm_train.m: Train Bradley-Terry model
	btmg_train.m: Train Bradley-Terry+G model proposed by Karthik Raman (http://www.cs.cornell.edu/people/tj/publications/raman_joachims_14a.pdf)
	rbtm_train.m: Train reffered Bradley-Terry model  proposed by Nihar B. Shah (http://lytics.stanford.edu/datadriveneducation/papers/shahetal.pdf)
	generate_Ordinal.m: Generate oridinal preferences from cardinal peer grading scores
	ordinal_eva.m: Compute ordinal evaluations of ordinal models

-------
scripts
-------
	PG3.m: Test PG3 individually
	PG4.m: Test PG4 individually
	PG5.m: Test PG5 individually
	main.m: Run both cardinal models (PG3, PG4, PG45) and cardinal+ordinal models together.

	Note: we didn't includes the hyperparameter tuning phase in our code, instead, we adopted the best settings for cardinal models, and corresponing settings for cardinal+ordinal models.