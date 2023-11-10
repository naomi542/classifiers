classdef Functions
    methods (Static = true)
    
        function matrix = generate_data(mean, covariance, n)
            matrix = repmat(mean,n,1) + randn(n,2)*chol(covariance);
        end
        
        function plot_samples_AB(A, B, mean_A, mean_B, evec_A, evec_B, eval_A, eval_B)
            % Plot samples of A and B
            scatter(A(:,1), A(:,2), 'filled', 'DisplayName', 'Class A', 'MarkerFaceColor', [171/255 111/255 169/255]);
            scatter(B(:,1), B(:,2), 'filled', 'DisplayName', 'Class B', 'MarkerFaceColor', [29/255 146/255 64/255]);

            % Plot standard deviation contours of A and B
            Functions.plot_ellipse(mean_A(1), mean_A(2), atan(evec_A(2,2)/evec_A(2,1)), sqrt(eval_A(2,2)), sqrt(eval_A(1,1)),'Contour A', [128/255 40/255 130/255]);
            Functions.plot_ellipse(mean_B(1), mean_B(2), atan(evec_B(2,2)/evec_B(2,1)), sqrt(eval_B(2,2)), sqrt(eval_B(1,1)),'Contour B', [11/255 82/255 30/255]);
        end
        
        function plot_samples_CDE(C, D, E, mean_C, mean_D, mean_E, evec_C, evec_D, evec_E, eval_C, eval_D, eval_E)
            % Plot samples of C, D, E
            scatter(C(:,1), C(:,2), 'filled', 'DisplayName', 'Class C','MarkerFaceColor', [171/255 111/255 169/255]);     
            scatter(D(:,1), D(:,2), 'filled', 'DisplayName', 'Class D','MarkerFaceColor', [29/255 146/255 64/255]);    
            scatter(E(:,1), E(:,2), 'filled', 'DisplayName', 'Class E','MarkerFaceColor', [0 76/255 153/255]);

            % Plot standard deviation contours
            Functions.plot_ellipse(mean_C(1), mean_C(2), atan(evec_C(2,2)/evec_C(2,1)), sqrt(eval_C(2,2)), sqrt(eval_C(1,1)), 'Contour C',[128/255 40/255 130/255]);
            Functions.plot_ellipse(mean_D(1), mean_D(2), atan(evec_D(2,2)/evec_D(2,1)), sqrt(eval_D(2,2)), sqrt(eval_D(1,1)), 'Contour D',[11/255 82/255 30/255]); 
            Functions.plot_ellipse(mean_E(1), mean_E(2), atan(evec_E(2,2)/evec_E(2,1)), sqrt(eval_E(2,2)), sqrt(eval_E(1,1)), 'Contour E',[44/255 85/255 132/255]);
        end
        
        function plot_ellipse(x,y,theta,a,b,disp_name,color)
            if nargin<5
                error('Too few arguments to Plot_Ellipse.'); 
            end

            np = 100;
            ang = [0:np]*2*pi/np;
            pts = [x;y]*ones(size(ang)) + [cos(theta) -sin(theta); sin(theta) cos(theta)]*[cos(ang)*a; sin(ang)*b];
            p = plot( pts(1,:), pts(2,:), 'r', 'LineWidth', 2 , 'DisplayName', disp_name,'Color', color);
            
        end
        
        function [class] = classify_point(X, Y, CD, CE, DE)
            class = zeros(size(CD));
            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    class(i, j) = Functions.determine_class(CD(i,j), CE(i,j), DE(i,j));
                end
            end
        end
        
        function point_class = determine_class (cd,ce,de)        
            if cd <= 0 && ce <= 0
                point_class= -1; %class C
            elseif de <= 0 && cd >= 0 
                point_class= 0; % class D
            elseif ce >= 0 && de >=0
                point_class= 1;  % class E
            end
        end
        
        function [MED] = get_MED(X, Y, mean1, mean2)
            MED = zeros(size(X,1), size(Y,2));

            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    point = [X(i,j) Y(i,j)];

                    % if < 0, belongs to class 1; if > 0, belongs to class 2
                    MED(i, j) = Functions.get_distance(point, mean1) - Functions.get_distance(point, mean2);
                end 
            end      
        end
        
        function dist = get_distance(x1, x2)
            dist = sqrt((x1-x2)*(x1-x2)');
        end

        function [GED] = get_GED(X, Y, mean1, mean2, cov1, cov2)
            GED = zeros(size(X,1), size(Y,2));
            
            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    x = [X(i, j) Y(i, j)];

                    % if < 0, belongs to class 1; if > 0, belongs to class 2
                    GED(i, j) = Functions.get_MICD_distance(x, cov1, mean1) - Functions.get_MICD_distance(x, cov2, mean2);
                end 
            end            
        end

        function dist = get_MICD_distance(x, covariance, mean)
            dist = (x-mean)*inv(covariance)*(x-mean)';
        end
    
        function [MAP] = get_MAP(X, Y, mean1, mean2, cov1, cov2, prior1, prior2)
            MAP = zeros(size(Y,1), size(X,2));

            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    x = [X(i,j) Y(i,j)];

                    % if > 0, belongs to class 1; if < 0, belongs to class 2
                    % log(x) is natural log
                    MICD_calc = Functions.get_MICD_distance(x, cov2, mean2) - Functions.get_MICD_distance(x, cov1, mean1);
                    MAP(i, j) = -1* (MICD_calc - 2*log(prior2 / prior1) - log(det(cov1) / det(cov2)));
                end 
            end
        end
        
        function [NN] = get_NN(X, Y, class1, class2)
            NN = zeros(size(Y, 1), size(X, 2));
            
            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    x = [X(i,j) Y(i,j)];
                    min1 = intmax; 
                    min2 = intmax;

                    % Find nearest neighbour for class 1
                    for k = 1:size(class1, 1)
                        d = Functions.get_distance(x, [class1(k, 1) class1(k, 2)]);
                        if d < min1, min1 = d; end
                    end
                    
                    % Find nearest neighbour for class 2
                    for k = 1:size(class2, 1)
                        d = Functions.get_distance(x, [class2(k, 1) class2(k, 2)]);
                        if d < min2, min2 = d; end
                    end
                    
                    NN(i, j) = min1-min2;
                end 
            end  
        end
        
        function [kNN] = get_kNN(X, Y, class1, class2, k)
            kNN = zeros(size(Y, 1), size(X, 2));
           
            for i = 1:size(X, 1)
                for j = 1:size(Y, 2)
                    x = [X(i,j) Y(i,j)];
                    min1 = zeros(k,3)+double(intmax);
                    min2 = zeros(k,3)+double(intmax);

                    % Find k nearest neighbours for class 1
                    for m = 1:size(class1, 1)
                        d = Functions.get_distance(x, [class1(m, 1) class1(m, 2)]);
                        [max_val, idx] = max(min1(:,1));
                        if d < max_val, min1(idx,1:3) = [d class1(m,1:2)];end
                    end
                  
                    % Find nearest neighbour for class 2
                    for m = 1:size(class2, 1)
                        d = Functions.get_distance(x, [class2(m, 1) class2(m, 2)]);
                        [max_val, idx] = max(min2(:,1));
                        if d < max_val, min2(idx,1:3) = [d class2(m,1:2)];end
                    end

                    % Find mean distance for each class
                    mean1 = mean(min1);
                    mean2 = mean(min2);

                    kNN(i,j) = Functions.get_distance(x, [mean1(2) mean1(3)])-Functions.get_distance(x, [mean2(2) mean2(3)]);                
                end
            end
        end
        
        %% Error Functions
        function [num_classified_class] = two_class_error(X, mean1, mean2,cov1,cov2,prior1,prior2)
            MED = zeros([size(X,1),1]);
            GED = zeros([size(X,1),1]);
            MAP = zeros([size(X,1),1]);
                
            for i = 1:size(X,1)
                point = X(i,:);
                MED(i) = Functions.get_distance(point, mean1) - Functions.get_distance(point, mean2);
                GED(i) = Functions.get_MICD_distance(point,cov1, mean1) - Functions.get_MICD_distance(point,cov2, mean2);
                MAP(i) = GED(i) + 2*log(prior2 / prior1) + log(det(cov1) / det(cov2));
            end
            
            %number classified as class1
            num_classified_class = [sum(MED < 0) sum(GED < 0) sum(MAP < 0)]; 
        end

        %% 3 Class MED, GED, MAP error
        function [num_classified_class] = three_class_error(class,mean1,mean2,mean3,cov1,cov2,cov3,prior1,prior2,prior3)
            %column1 = class 1 and 2, column2 = class 1 and 3, column3 = class 2 and 3 
            MED = zeros(size(class,1),3);
            GED = zeros(size(class,1),3);
            MAP = zeros(size(class,1),3);
    
            for i = 1:size(class,1)
                point = class(i,:);
                MED(i,:) = [Functions.get_distance(point, mean1) - Functions.get_distance(point, mean2);
                            Functions.get_distance(point, mean1) - Functions.get_distance(point, mean3);
                            Functions.get_distance(point, mean2) - Functions.get_distance(point, mean3)];
                        
                GED(i,:) = [Functions.get_MICD_distance(point,cov1, mean1) - Functions.get_MICD_distance(point,cov2, mean2);
                            Functions.get_MICD_distance(point,cov1, mean1) - Functions.get_MICD_distance(point,cov3, mean3);
                            Functions.get_MICD_distance(point,cov2, mean2) - Functions.get_MICD_distance(point,cov3, mean3)];
                            
                MAP(i,:) = [GED(i,1) + 2*log(prior2 / prior1) + log(det(cov1) / det(cov2));
                            GED(i,2) + 2*log(prior3 / prior1) + log(det(cov1) / det(cov3));
                            GED(i,3) + 2*log(prior3 / prior2) + log(det(cov2) / det(cov3))];
            end
            
            MED_classified = zeros(size(class,1),1);
            GED_classified = zeros(size(class,1),1);
            MAP_classified = zeros(size(class,1),1);
            
            for i = 1:size(class,1)
                MED_classified(i) = Functions.determine_class(MED(i,1), MED(i,2), MED(i,3));
                GED_classified(i) = Functions.determine_class(GED(i,1), GED(i,2), GED(i,3));
                MAP_classified(i) = Functions.determine_class(MAP(i,1), MAP(i,2), MAP(i,3));
            end
            
            % numbers for MED, GED and MAP
            num_classified_class = [sum(MED_classified==[-1,0,1]);sum(GED_classified==[-1,0,1]);sum(MAP_classified== [-1,0,1])];
        end
        
        %% 2 Class NN, kNN Error
        function [num_classified_class] = two_class_error_NN_kNN(class, class1, class2)
            class_col2 = class(:,2);
            NN = Functions.get_NN(class, class_col2, class1, class2);
            kNN = Functions.get_kNN(class, class_col2, class1, class2, 5);
            
            %number classified as class A
            num_classified_class(1) = sum(NN(:,1) < 0); 
            num_classified_class(2) = sum(kNN(:,1) < 0);
        end
        
        %% 3 Class NN, kNN Error
        function [num_classified_class] = three_class_error_NN_kNN(class, class1, class2, class3, k)
            NN = zeros(size(class, 1),6);
            kNN = zeros(size(class, 1),6);
            class_col2 = class(:,2);
            
            % C and D
            NN(:,1:2) = Functions.get_NN(class, class_col2, class1, class2);
            kNN(:,1:2) = Functions.get_kNN(class, class_col2, class1, class2, k);
            
            % C and E
            NN(:,3:4) = Functions.get_NN(class, class_col2, class1, class3);
            kNN(:,3:4) = Functions.get_kNN(class, class_col2, class1, class3, k);
            
            % D and E
            NN(:,5:6) = Functions.get_NN(class, class_col2, class2, class3);
            kNN(:,5:6) = Functions.get_kNN(class, class_col2, class2, class3, k);
            
            NN_classified = zeros(size(class,1),1);
            kNN_classified = zeros(size(class,1),1);
            
            for i = 1:size(class,1)
                NN_classified(i) = Functions.determine_class(NN(i,1), NN(i,3), NN(i,5));
                kNN_classified(i) = Functions.determine_class(kNN(i,1), kNN(i,3), kNN(i,5));
            end
            
            %number classified as class A
            num_classified_class = [sum(NN_classified==[-1,0,1]) ; sum(kNN_classified==[-1,0,1])];
        end
    end
end