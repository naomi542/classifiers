clc;
close all;
clear all; 
rng(1);

mymap = [224/255 173/255 255/255
         204/255 255/255 204/255
         161/255 221/255 230/255];
     
%% Initializing Sample Data and Statistics

mean_A = [5 10];
mean_B = [10 15];
mean_C = [5 10];
mean_D = [15 10];
mean_E = [10 5];

cov_A = [8 0; 0 4];
cov_B = [8 0; 0 4];
cov_C = [8 4; 4 40];
cov_D = [8 0; 0 8];
cov_E = [10 -5; -5 20];

[evec_A, eval_A] = eig(cov_A);
[evec_B, eval_B] = eig(cov_B);
[evec_C, eval_C] = eig(cov_C);
[evec_D, eval_D] = eig(cov_D);
[evec_E, eval_E] = eig(cov_E);

n_A = 200;
n_B = 200;
n_C = 100;
n_D = 200;
n_E =150;

A = Functions.generate_data(mean_A, cov_A, n_A);
B = Functions.generate_data(mean_B, cov_B, n_B);
C = Functions.generate_data(mean_C, cov_C, n_C);
D = Functions.generate_data(mean_D, cov_D, n_D);
E = Functions.generate_data(mean_E, cov_E, n_E);

%% Creating a Grid of Points

increment = 0.1; % The lower this is the smoother the contours.

% For classes A an B
x1 = min([A(:,1);B(:,1)]):increment:max([A(:,1);B(:,1)]);
y1 = min([A(:,2);B(:,2)]):increment:max([A(:,2);B(:,2)]);
[X1, Y1] = meshgrid(x1,y1);

% For classes C, D, E
x2 = min([C(:,1);D(:,1);E(:,1)]):increment:max([C(:,1);D(:,1);E(:,1)]);
y2 = min([C(:,2);D(:,2);E(:,2)]):increment:max([C(:,2);D(:,2);E(:,2)]);
[X2, Y2] = meshgrid(x2,y2);

%% Standard Deviation Contours

% Classes A and B
figure('Name','Classes A and B');
set(gcf,'color','w');
hold on

Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

% Classes C, D, E
figure('Name','Classes C,D,E');
set(gcf,'color','w');
hold on

Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% MED for Classes A and B

% Calculate MED
MED_AB = Functions.get_MED(X1, Y1, mean_A, mean_B);

% Plot samples, std contours, and MED decision boundary
figure('Name','MED A and B');
set(gcf,'color','w');
hold on

contourf(X1, Y1, MED_AB, [0,0], 'k', 'LineWidth', 2, 'DisplayName', 'MED Decision Boundary');
colormap(mymap)
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

%% MED for Classes C, D, E

% Compute MED boundaries
MED_CD = Functions.get_MED(X2, Y2, mean_C, mean_D);
MED_CE = Functions.get_MED(X2, Y2, mean_C, mean_E);
MED_DE = Functions.get_MED(X2, Y2, mean_D, mean_E);
MED_CDE = Functions.classify_point(X2, Y2, MED_CD, MED_CE, MED_DE);

% Plot samples, std contours, MED decision boundaries
figure('Name','MED C,D,E');
set(gcf,'color','w');
hold on

contourf(X2, Y2, MED_CDE, [-1 0 1], 'k', 'LineWidth', 2, 'DisplayName', 'MED Decision Boundary');
colormap(mymap);
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% GED for Classes A and B

% Compute GED decision boundary
GED_AB = Functions.get_GED(X1, Y1, mean_A, mean_B, cov_A, cov_B);

% Plot samples, std contours, GED decision boundary
figure('Name','GED A and B');
set(gcf,'color','w');
hold on

contourf(X1, Y1, GED_AB, [0,0], 'k', 'LineWidth', 2, 'DisplayName', 'GED Decision Boundary');
colormap(mymap)
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

%% GED for Classes C, D, E

% GED/MICD boundaries
GED_CD = Functions.get_GED(X2, Y2, mean_C, mean_D, cov_C, cov_D);
GED_CE = Functions.get_GED(X2, Y2, mean_C, mean_E, cov_C, cov_E);
GED_DE = Functions.get_GED(X2, Y2, mean_D, mean_E, cov_D, cov_E);

GED_CDE = Functions.classify_point(X2, Y2, GED_CD, GED_CE, GED_DE);

% Plot samples, std contours, GED/MICD decision boundary
figure('Name','GED C,D,E');
set(gcf,'color','w');
hold on

contourf(X2, Y2, GED_CDE, [-1 0 1], 'k', 'LineWidth', 2, 'DisplayName', 'GED Decision Boundary');
colormap(mymap)
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% MAP for Classes A and B
prior_A = n_A/(n_A + n_B);
prior_B = 1-prior_A;

MAP_AB = Functions.get_MAP(X1, Y1, mean_A, mean_B, cov_A, cov_B, prior_A, prior_B);

% Plot samples, std contours, MAP decision boundary
figure('Name','MAP A and B');
set(gcf,'color','w');
hold on

contourf(X1, Y1, MAP_AB, [0 0], 'k', 'LineWidth', 2, 'DisplayName', 'MAP Decision Boundary');
colormap(mymap)
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

%% MAP for Classes C, D, E
prior_C = n_C / (n_C + n_D + n_E);
prior_D = n_D / (n_C + n_D + n_E);
prior_E = n_E / (n_C + n_D + n_E);

MAP_CD = Functions.get_MAP(X2, Y2, mean_C, mean_D, cov_C, cov_D, prior_C, prior_D);
MAP_CE = Functions.get_MAP(X2, Y2, mean_C, mean_E, cov_C, cov_E, prior_C, prior_E);
MAP_DE = Functions.get_MAP(X2, Y2, mean_D, mean_E, cov_D, cov_E, prior_D, prior_E);
MAP_CDE = Functions.classify_point(X2, Y2, MAP_CD, MAP_CE, MAP_DE);

% Plot samples, std contours, MAP decision boundary
figure('Name','MAP C,D,E');
set(gcf,'color','w');
hold on

contourf(X2, Y2, MAP_CDE, [-1,-0,1], 'k', 'LineWidth', 2, 'DisplayName', 'MAP Decision Boundary');
colormap(mymap)
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% NN for Classes A and B

% Compute NN decision boundary
NN_AB = Functions.get_NN(X1, Y1, A, B);

% Plot samples, std contours, NN decision boundary
figure('Name','NN A and B');
set(gcf,'color','w');
hold on

contourf(X1, Y1, NN_AB, [0,0], 'k', 'LineWidth', 2, 'DisplayName', 'NN Decision Boundary');
colormap(mymap)
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

%% NN for Classes C, D, E

% Compute NN decision boundary
NN_CD = Functions.get_NN(X2, Y2, C, D);
NN_CE = Functions.get_NN(X2, Y2, C, E);
NN_DE = Functions.get_NN(X2, Y2, D, E);
NN_CDE = Functions.classify_point(X2, Y2, NN_CD, NN_CE, NN_DE);

% Plot samples, std contours, NN decision boundary
figure('Name','NN C,D,E');
set(gcf,'color','w');
hold on

contourf(X2, Y2, NN_CDE, [-1,0,1], 'k', 'LineWidth', 2, 'DisplayName', 'NN Decision Boundary');
colormap(mymap)
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% kNN for Classes A and B

% Compute kNN decision boundary
kNN_AB = Functions.get_kNN(X1, Y1, A, B, 5);

% Plot samples, std contours, NN decision boundary
figure('Name','kNN A and B');
set(gcf,'color','w');
hold on

contourf(X1, Y1, kNN_AB, [0,0], 'k', 'LineWidth', 2, 'DisplayName', 'kNN Decision Boundary');
colormap(mymap)
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

%% kNN for Classes C, D, E

% Compute NN decision boundary
kNN_CD = Functions.get_kNN(X2, Y2, C, D,5);
kNN_CE = Functions.get_kNN(X2, Y2, C, E,5);
kNN_DE = Functions.get_kNN(X2, Y2, D, E,5);
kNN_CDE = Functions.classify_point(X2, Y2, kNN_CD, kNN_CE, kNN_DE);

% Plot samples, std contours, NN decision boundary
figure('Name','KNN C,D,E');
set(gcf,'color','w');
hold on

contourf(X2, Y2, kNN_CDE, [-1,0,1], 'k', 'LineWidth', 2, 'DisplayName', 'kNN Decision Boundary');
colormap(mymap)
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off


%% Plotting MED, GED/MICD, MAP Together

% Case 1: Classes A and B
figure();
set(gcf,'color','w');
hold on

contour(X1, Y1, MED_AB, [0,0], 'r', 'LineWidth', 2, 'DisplayName', 'MED Decision Boundary');
contour(X1, Y1, GED_AB, [0,0], 'b', 'LineWidth', 2, 'DisplayName', 'MICD Decision Boundary');
contour(X1, Y1, MAP_AB, [0 0], 'k', 'LineWidth', 2, 'DisplayName', 'MAP Decision Boundary');
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

% Case 2: Classes C, D, E
figure();
set(gcf,'color','w');
hold on

contour(X2, Y2, MED_CDE, [-1,0,1], 'r', 'LineWidth', 2, 'DisplayName', 'MED Decision Boundary');
contour(X2, Y2, GED_CDE, [-1 0 1], 'b', 'LineWidth', 2, 'DisplayName', 'MICD Decision Boundary');
contour(X2, Y2, MAP_CDE, [-1,0,1], 'k', 'LineWidth', 2, 'DisplayName', 'MAP Decision Boundary');
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% Plotting NN, kNN Together

% Case 1: Classes A and B
figure();
set(gcf,'color','w');
hold on

contour(X1, Y1, NN_AB, [0,0], 'r', 'LineWidth', 2, 'DisplayName', 'NN Decision Boundary');
contour(X1, Y1, kNN_AB, [0,0], 'k', 'LineWidth', 2, 'DisplayName', '5NN Decision Boundary');
Functions.plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B);

legend
hold off

% Case 2: Classes C, D, E
figure();
set(gcf,'color','w');
hold on

contour(X2, Y2, NN_CDE, [-1,0,1], 'r', 'LineWidth', 2, 'DisplayName', 'NN Decision Boundary');
contour(X2, Y2, kNN_CDE, [-1,0,1], 'k', 'LineWidth', 2, 'DisplayName', '5NN Decision Boundary');
Functions.plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E);

legend
hold off

%% Error Analysis for MED, GED, MAP
 
% 2 Class Error and Confusion Matrices
results_A = Functions.two_class_error(A, mean_A, mean_B, cov_A, cov_B, prior_A, prior_B);
results_B = n_B - Functions.two_class_error(B, mean_A, mean_B, cov_A, cov_B, prior_A, prior_B);
 
MED_error_A = (n_A-results_A(1))/n_A;
MED_error_B = (n_B-results_B(1))/n_B;
MED_error_AB = ((n_A-results_A(1)) + (n_B-results_B(1))) / (n_A + n_B);
MED_confusion_AB = [results_A(1),n_A-results_A(1);n_B-results_B(1),results_B(1)]
 
GED_error_A = (n_A-results_A(2))/n_A;
GED_error_B = (n_B-results_B(2))/n_B;
GED_error_AB = ((n_A-results_A(2)) + (n_B-results_B(2))) / (n_A + n_B);
GED_confusion_AB = [results_A(2),n_A-results_A(2);n_B-results_B(2),results_B(2)]
 
MAP_error_A = (n_A-results_A(3))/n_A;
MAP_error_B = (n_B-results_B(3))/n_B;
MAP_error_AB = ((n_A-results_A(3)) + (n_B-results_B(3))) / (n_A + n_B);
MAP_confusion_AB = [results_A(3),n_A-results_A(3);n_B-results_B(3),results_B(3)]
 
% 3 Class Error and Confusion Matrices
results_C = Functions.three_class_error(C, mean_C, mean_D, mean_E, cov_C, cov_D, cov_E, prior_C, prior_D, prior_E);
results_D = Functions.three_class_error(D, mean_C, mean_D, mean_E, cov_C, cov_D, cov_E, prior_C, prior_D, prior_E);
results_E = Functions.three_class_error(E, mean_C, mean_D, mean_E, cov_C, cov_D, cov_E, prior_C, prior_D, prior_E);
 
MED_error_C = (n_C-results_C(1,1))/n_C;
MED_error_D = (n_D-results_D(1,2))/n_D;
MED_error_E = (n_E-results_E(1,3))/n_E;
MED_error_CDE = ((n_C-results_C(1,1)) + (n_D-results_D(1,2)) + (n_E-results_E(1,3))) / (n_C + n_D + n_E);
MED_confusion_CDE = [results_C(1,:);results_D(1,:);results_E(1,:)]
 
GED_error_C = (n_C-results_C(2,1))/n_C;
GED_error_D = (n_D-results_D(2,2))/n_D;
GED_error_E = (n_E-results_E(2,3))/n_E;
GED_error_CDE = ((n_C-results_C(2,1)) + (n_D-results_D(2,2)) + (n_E-results_E(2,3))) / (n_C + n_D + n_E);
GED_confusion_CDE = [results_C(2,:);results_D(2,:);results_E(2,:)]
 
MAP_error_C = (n_C-results_C(3,1))/n_C;
MAP_error_D = (n_D-results_D(3,2))/n_D;
MAP_error_E = (n_E-results_E(3,3))/n_E;
MAP_error_CDE = ((n_C-results_C(3,1)) + (n_D-results_D(3,2)) + (n_E-results_E(3,3))) / (n_C + n_D + n_E);
MAP_confusion_CDE = [results_C(3,:);results_D(3,:);results_E(3,:)]

%% Error Analysis for NN and kNN
 
% Test Samples for NN/kNN Error Analysis
rng(2);
Test_A = Functions.generate_data(mean_A, cov_A, n_A);
Test_B = Functions.generate_data(mean_B, cov_B, n_B);
Test_C = Functions.generate_data(mean_C, cov_C, n_C);
Test_D = Functions.generate_data(mean_D, cov_D, n_D);
Test_E = Functions.generate_data(mean_E, cov_E, n_E);
 
% Case 1: 2 Classes
results_A = Functions.two_class_error_NN_kNN(Test_A, A, B);
results_B = n_B - Functions.two_class_error_NN_kNN(Test_B, A, B);
 
NN_error_A = (n_A-results_A(1))/n_A;
NN_error_B = (n_B-results_B(1))/n_B;
NN_error_AB = ((n_A-results_A(1)) + (n_B-results_B(1))) / (n_A + n_B);
NN_confusion_AB = [results_A(1),n_A-results_A(1);n_B-results_B(1),results_B(1)]
 
kNN_error_A = (n_A-results_A(2))/n_A;
kNN_error_B = (n_B-results_B(2))/n_B;
kNN_error_AB = ((n_A-results_A(2)) + (n_B-results_B(2))) / (n_A + n_B);
kNN_confusion_AB = [results_A(2),n_A-results_A(2);n_B-results_B(2),results_B(2)]
 
% Case 2: 3 Classes
results_C = Functions.three_class_error_NN_kNN(Test_C, C, D, E, 5);
results_D = Functions.three_class_error_NN_kNN(Test_D, C, D, E, 5);
results_E = Functions.three_class_error_NN_kNN(Test_E, C, D, E, 5);
 
NN_error_C = (n_C-results_C(1,1))/n_C;
NN_error_D = (n_D-results_D(1,2))/n_D;
NN_error_E = (n_E-results_E(1,3))/n_E;
NN_error_CDE = ((n_C-results_C(1,1)) + (n_D-results_D(1,2)) + (n_E-results_E(1,3))) / (n_C + n_D + n_E);
NN_confusion_CDE = [results_C(1,:);results_D(1,:);results_E(1,:)]
 
kNN_error_C = (n_C-results_C(2,1))/n_C;
kNN_error_D = (n_D-results_D(2,2))/n_D;
kNN_error_E = (n_E-results_E(2,3))/n_E;
kNN_error_CDE = ((n_C-results_C(2,1)) + (n_D-results_D(2,2)) + (n_E-results_E(2,3))) / (n_C + n_D + n_E);
kNN_confusion_CDE = [results_C(2,:);results_D(2,:);results_E(2,:)]