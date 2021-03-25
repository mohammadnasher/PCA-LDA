%   Author  : Mohammad Nasher
%   Date    : 18 March 2021
%   Based on Matlab 2013a Features
%   Contact : nasher[at]iasbs[dot]ac[dot]ir
%   Developed for Final Project of Computational Data Mining Course which
%   Defined to Analyze Gene Data Set using PCA and LDA

clc
close all
clear all

%   Change the PLOT_MODE Variable 
PLOT_MODE          = true;
LUMINAL_VALUE      = 'luminal';
NON_LUMINAL_VALUE  = 'non-luminal';
NUM_OF_LUM         = 1;
NUM_OF_NON_LUM     = 1;
EMPTY_DATA         = [];
DATA_SIZE          = 128;
                
%   Question #1 : Appliment of PCA using SVD

%   Step1: Read Data from Specified .xslx File
[DATA_R, DATA_TAG, DATA_TMP] = xlsread('breast_preprocessed.xlsx');
%   Step2: Pre-Process Data
%   to Pre-Process Data We should Do 2 Major Sub-Steps:
%   Sub-Step 1: Remove Spam Rows
TAG    = DATA_TAG(end , 2:end);
DATA   = DATA_R(1:end-2 , 1:end);
%   We can Visualize Pre-Processed Data As a Figure for SubStep1
if PLOT_MODE
    figure;
    plot(DATA,'.');
    title('DATA AFTER SUBSTEP1(TAG_REMOVAL)');
end
%   SubStep 2: As the Problem Definition, Mean-Center Progress 
%   is the Second Sub-Step
DATA_MEAN       = mean(DATA, 2);
for INDEX = 1: DATA_SIZE
    NORMALIZED_DATA(:, INDEX) = DATA(:, INDEX) - DATA_MEAN;
end
%   We can Visualize Pre-Processed Data As a Figure for SubStep2
if PLOT_MODE
    figure;
    plot(NORMALIZED_DATA,'.');
    title('DATA AFTER SUBSTEP2(MEAN_CENTER)');
end
%   Step3: Apply PCA using SVD 
%   The Output of SVD is 3 Different Matrixes as U, S, V
[U_Matrix, S_Matrix, V_Matrix] = svd(NORMALIZED_DATA,'econ');

%   as the SVD Decomposition for PCA we can Determine Loading Matrix and 
%   Score Matrix as Below
LOADING_MATRIX = U_Matrix;
SCORE_MATRIX   = S_Matrix*V_Matrix';
RD_DATA        = U_Matrix * S_Matrix;

%   Step4: Feature Seperation using ScoreMatrix
for INDEX = 1:size(SCORE_MATRIX)
    %   If the Selected Index is Non Luminal Provide Appropriate Action 
    %   On the Selected Score Matrix Dimention
    if(isequal(NON_LUMINAL_VALUE, char(TAG(INDEX))))
        NON_LUMINALS(:,NUM_OF_NON_LUM) = SCORE_MATRIX(:,INDEX);
        %   Incrementally Count the number of Non Luminals
        NUM_OF_NON_LUM = NUM_OF_NON_LUM + 1;
    %   If the Selected Index is Luminal Provide Appropriate Action 
    %   On the Selected Score Matrix Dimention
    else
        LUMINALS(:,NUM_OF_LUM) = SCORE_MATRIX(:,INDEX);
        %   Incrementally Count the number of Luminals
        NUM_OF_LUM = NUM_OF_LUM + 1;
    end
end

%   Step5: Variance and Comulative Frequency
%   Compute Variance using Features of S Matrix on main diagon
VARIANCE = 	zeros(size(S_Matrix),size(S_Matrix));   
for INDEX = 1 : size(S_Matrix)
    %   Diagonal Elements Will be Considered
    VARIANCE(INDEX) = S_Matrix(INDEX,INDEX);
end

CM_FREQUENCY   = zeros(size(VARIANCE'))
%   At the First Step the Cumulative Frequency is equal to Variance
CM_FREQUENCY(1)= VARIANCE(1);
%   For each Index will Compute the Cumulative Frequency as The Specified 
%   Equation
for INDEX = 2 : size(VARIANCE')
    CM_FREQUENCY(INDEX) = CM_FREQUENCY(INDEX - 1) + VARIANCE(INDEX);
end
%   Finally we Plot the Commulative Variace of Data as below
if PLOT_MODE
    figure;
    bar(CM_FREQUENCY);
    title('CUMULATIVE FREQUENCY PLOT');
end

%   Step6: Create and Show Loading Plot using Loading Matrix Values
if PLOT_MODE
    figure;
    plotv(LOADING_MATRIX)
    title('LOADING PLOT')
end
%   Step7:  Create and Show Scroe Plot using Luminal and Non-Luminal 
%   Data Seperation which we Have Done on Step4
if PLOT_MODE
    figure;
    plot(LUMINALS,'ro');
    hold on
    plot(NON_LUMINALS,'bs');
    legend('LUMINAL','NON-LUMINAL');
    title('SCORE PLOT')
end        


%   Question #2 : Appliment of LDA
CL_LUMINAL     = [];
CL_NON_LUMINAL = [];
%   use SVD Decomposition Features to Produce to Matrix Classes
for INDEX = 1:DATA_SIZE
    if isequal(LUMINAL_VALUE, char(TAG(INDEX)))
        CL_LUMINAL     = [CL_LUMINAL; RD_DATA(INDEX,:)];
    else
        CL_NON_LUMINAL = [CL_NON_LUMINAL; RD_DATA(INDEX,:)];
    end
end
%   As the Project Definition Document We Should Use 14 Data for Each 
%   Category of Data Labels
LUMINAL_TEST     = CL_LUMINAL(size(CL_LUMINAL, 1) - 13 : size(CL_LUMINAL, 1),:);
NON_LUMINAL_TEST = CL_NON_LUMINAL(size(CL_NON_LUMINAL) - 13 : size(CL_NON_LUMINAL),:); 
%   To Update the Base Classes for Test Set We Set Each Chosen Row to Null
CL_LUMINAL(size(CL_LUMINAL) - 13 : size(CL_LUMINAL),:)             = [];
CL_NON_LUMINAL(size(CL_NON_LUMINAL) - 13 : size(CL_NON_LUMINAL),:) = [];

%   Step2: Get Mean of Data  
LUMINAL_MEAN     = mean(CL_LUMINAL, 1);
NON_LUMINAL_MEAN = mean(CL_NON_LUMINAL, 1);

%   Step3: SW Calculation 
SI_1 = zeros(DATA_SIZE ,DATA_SIZE);
for INDEX = 1 : size(CL_LUMINAL)
    SI_1 = SI_1 + (CL_LUMINAL(INDEX,:)- LUMINAL_MEAN)' * (CL_LUMINAL(INDEX,:) - LUMINAL_MEAN);
end
SI_2 = zeros(DATA_SIZE,DATA_SIZE);
for INDEX = 1 : size(CL_NON_LUMINAL, 1)
    SI_2 = SI_2+ (CL_NON_LUMINAL(INDEX,:) - NON_LUMINAL_MEAN)' * (CL_NON_LUMINAL(INDEX,:) - NON_LUMINAL_MEAN);
end     
%   SW Computation
SW = SI_1 + SI_2;
%   Compute Total Mean to Help Computation of Sb
TOTAL_MEAN = mean([CL_LUMINAL; CL_NON_LUMINAL]);
S_B = zeros(DATA_SIZE,DATA_SIZE);
S_B = S_B + size(CL_LUMINAL,1) * (LUMINAL_MEAN - TOTAL_MEAN)' * (LUMINAL_MEAN - TOTAL_MEAN);
S_B = S_B + size(CL_NON_LUMINAL,1) * (NON_LUMINAL_MEAN - TOTAL_MEAN)' * (NON_LUMINAL_MEAN - TOTAL_MEAN);

%   Since the SW in not diagonal get the pseudo inverse of it using 
%   Moore-Penrose pseudoinverse function
INV_SW     = pinv(SW);
S_Matrix   = INV_SW * S_B;
%   Do extraction of eigen values from S_Matrix
[V_m, D_m] = eig(S_Matrix);
%   Diagonalization of eigen outputs
EIGEN_VALS = diag(D_m);
%   In the Semi-Last Step We Must Sort Eigen Vectors Decreasing way
%   Hold Number of Eigen Vectorsin W_Matrix
W_SIZE    = size(V_m, 2);
W_Matrix  = zeros(W_SIZE, W_SIZE);   
for INDEX = 1:size(V_m, 2)
    %   Create Eigen Vector of The Max Eigen Value
    W_Matrix(:, INDEX)                         = V_m(:,D_m(INDEX) == max(D_m(INDEX)));  
    V_m(:,D_m(INDEX) == max(D_m(INDEX)))       = [];
    D_m(D_m(INDEX) == max(D_m(INDEX)))         = [];
end
%   On the Last Step We Create a Multipilication Value of Eigen Matrix W
%   and Luminal and Non-Luminal Collections to Create Final Results
F_LUMINAL        = CL_LUMINAL * W_Matrix;
F_NONN_LUMINAL   = CL_NON_LUMINAL * W_Matrix;

%   Test LDA Prediction on Reserved Test Cases
%   Find the Mean Value of Finalized Data 
LUMINAL_MEAN     = mean(F_LUMINAL ,1);
NON_LUMINAL_MEAN = mean(F_NONN_LUMINAL ,1);


%   Set Prediction Result Counter to Zero
CR_PREDICT = 0;
%   For each Test Cases Determine the Distance with Each Cluster 
%   If the Distance Agreed with Current Data Label We had a new Correct
%   Prediction using LDA on Luminal Classes
for INDEX = 1 : size(LUMINAL_TEST ,1)
    %    Distance Check
   if sqrt(sum((LUMINAL_MEAN - LUMINAL_TEST(INDEX, :)).^2, 2)) <  sqrt(sum((NON_LUMINAL_MEAN - LUMINAL_TEST(INDEX, :)).^2, 2))
        CR_PREDICT = CR_PREDICT + 1;
   end
end
%   For each Test Cases Determine the Distance with Each Cluster 
%   If the Distance Agreed with Current Data Label We had a new Correct
%   Prediction using LDA on Non Luminal Classes
for INDEX = 1 : size(NON_LUMINAL_TEST,1)
   %    Distance Check
   if sqrt(sum((NON_LUMINAL_MEAN - NON_LUMINAL_TEST(INDEX, :)).^2, 2)) <  sqrt(sum((LUMINAL_MEAN - NON_LUMINAL_TEST(INDEX, :)).^2, 2))
        CR_PREDICT = CR_PREDICT + 1;
   end
end




































        
        
        
        
        
        
