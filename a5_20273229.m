function a5_20273229
% Function for CISC271, Winter 2022, Assignment #5

    % Read the test data from a CSV file, standardize, and extract labels;
    % for the "college" data and Fisher's Iris data

    Xraw = csvread('collegenum.csv',1,1);
    [~, Xcoll] = pca(zscore(Xraw(:,2:end)), 'NumComponents', 2);
    ycoll = round(Xraw(:,1) > 0);

    load fisheriris;
    Xiris = zscore(meas);
    yiris = ismember(species,'setosa');

    % Call the functions for the questions in the assignment
    a5q1(Xcoll, ycoll);
    a5q2(Xiris, yiris);
    
% END FUNCTION
end

function a5q1(Xmat, yvec)
% A5Q1(XMAT,YVEC) solves Question 1 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as >=0 or <0
% OUTPUTS:
%         none

    % Augment the X matrix with a 1's vector
    Xaug = [Xmat ones(size(Xmat, 1), 1)];

    % Perceptron initialization and estimate of hyperplane
    eta = 0.001;
    [v_ann ix] = sepbinary(Xaug, yvec, eta);
    v_ann = v_ann/norm(v_ann);
    
    % Logistic regression estimate of hyperplane
    v_log = logreg(Xmat, yvec);
    v_log = v_log/norm(v_log);

    % Score the data using the hyperplane augmented vectors
    z_ann = Xaug*v_ann;
    z_log = Xaug*v_log;
    
    % Find the ROC curves
    [px_ann, py_ann, ~, auc_ann, t_ann] = perfcurve(yvec, z_ann, +1);
    [px_log, py_log, ~, auc_log, t_log] = perfcurve(yvec, z_log, +1);
 
    % %
    % % STUDENT CODE GOES HERE: compute the accuracies of hyperplanes
    % % and display the results to the command console
    % %

    % Compute the accuracies of the hyperplanes
    % First convert predictions to logical arrays
    y_pred_ann = z_ann >= 0;
    y_pred_log = z_log >= 0;
    % Computes number of correct predictions divided by number of
    % observations
    acc_ann = sum(y_pred_ann == yvec)/length(yvec)
    acc_log = sum(y_pred_log == yvec)/length(yvec)

    % Optional: finding the optimal threshold based on best accuracy
    t_opt_ann = -inf;
    acc_opt_ann = -inf;
    t_opt_log = -inf;
    acc_opt_log = -inf;
    for i=1:length(yvec)
 
        acc_temp_ann = (px_ann(i) * sum(yvec >= 0) + (1 - py_ann(i)) * sum(yvec < 0)) / length(yvec);
        if acc_temp_ann > acc_opt_ann
            acc_opt_ann = acc_temp_ann;
            t_opt_ann = z_ann(i);
        end
        
        acc_temp_log = (px_log(i) * sum(yvec >= 0) + (1 - py_log(i)) * sum(yvec < 0)) / length(yvec);
        if acc_temp_log > acc_opt_log
            acc_opt_log = acc_temp_log;
            t_opt_log = z_log(i);
        end
    end
    %{
    disp('Optimal threshold and accuracy for hyperplane found using Perceptron algorithm')
    disp(t_opt_ann)
    disp(acc_opt_ann)
    disp(t_ann)
    disp('Optimal threshold and accuracy for hyperplane found using logsitic regression')
    disp(t_opt_log)
    disp(acc_opt_log)
    disp(t_log)
    %}
    
    
    % %
    % % STUDENT CODE GOES HERE: plot figures and display results
    % %
    % ROC curves
    figure(1);
    plot(px_ann, py_ann),
    title('ROC Curve for Hyperplane Found by the Perceptron Rule')
    xlabel('False Positive Rate'),
    ylabel('True Positive Rate'),
    text(0.6, 0.4, 'AUC = 0.9495', 'FontSize', 14);

    figure(2);
    plot(px_log, py_log),
    title('ROC Curve for Hyperplane Found Using Logistic Regression')
    xlabel('False Positive Rate'),
    ylabel('True Positive Rate'),
    text(0.6, 0.4, 'AUC = 0.9600', 'FontSize', 14);

    % Scatter plots and separating lines
    figure(3);
    gscatter(Xmat(:,1), Xmat(:,2), yvec, 'rb'),
    title('Visualization of Hyperlane Found by the Perpcetron Rule'),
    xlabel('Loading Vector 1'),
    ylabel('Loading Vector 2'),
    plotline(v_ann, 'g');
    
    figure(4);
    gscatter(Xmat(:,1), Xmat(:,2), yvec, 'rb'),
    title('Visualization of Hyperlane Found Using Logistic Regression'),
    xlabel('Loading Vector 1'),
    ylabel('Loading Vector 2'),
    plotline(v_log, 'p');

% END FUNCTION
end

function [v_final, i_used] = sepbinary(Xmat, yvec, eta_in)
% [V_FINAL,I_USED]=SEPBINARY(XMAT,YVEC,ETA_IN)
% uses the Percetron Algorithm with a learning rate to linearly separate training vectors
% INPUTS:
%         XMAT    - Mx(N+1) augmented data matrix
%         YVEC    - Mx1 desired classes, 0 or 1
%         ETA_IN  - optional, scalar learning rate, default is 1
% OUTPUTS:
%         V_FINAL - (N+1)-D new estimated weight vector
%         I_USED  - scalar number of iterations used
% ALGORITHM:
%         Vectorized form of perceptron gradient descent

    % Use optional argument if it is preset
    if nargin>=3 && exist('eta_in') && ~isempty(eta_in)
        eta = eta_in;
    else
        eta = 1;
    end
    
    % Internal constant: maximum iterations to use
    imax = 10000;

    % Initialize the augmented weight vector as a 1's vector
    v_est = ones(size(Xmat, 2), 1);

    % Loop a limited number of times
    for i_used=0:imax

        % Assume that the final weight is the current estimate
        v_final = v_est;
 
        % Score the current estimate of the weight vector
        %z = Xmat * v_est;
        
        % Compute the predicted classification
        y_pred = Xmat*v_est>=0;
 
        % Compute the residual error (difference between predicted and actual class)
        rvec = yvec - y_pred;
 
        % Update based on error val using the learning rate eta
        v_est = v_est + eta * Xmat' * rvec;
 
        % Continue looping if any data are mis-classified
        missed = norm(rvec, 1) > 0;
 
        % Stop if the current estimate has converged
        if (missed==0)
            v_final = v_est;
            break;
        end
    end

% END FUNCTION
end


function a5q2(Xmat, yvec)
% A5Q2(XMAT,YVEC) solves Question 2 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as ~=0 or ==0
% OUTPUTS:
%         none

    % Anonymous function: centering matrix of parameterized size
    Gmat =@(k) eye(k) - 1/k*ones(k,k);

    % Problem size
    [m, n] = size(Xmat);

    % Default projection of data
    %Mgram = Xmat(:, 1:2);

    % Reduce data to Lmax-D; here, to 2D
    Lmax = 2;

    % Set an appropriate gamma for a Gaussian kernel
    sigma2 = 2*m;

    % Compute the centered MxM Gram matrix for the data
    % Using kernel trick to avoid embedding, improves efficiency
    Kmat = Gmat(m)*gramgauss(Xmat, sigma2)*Gmat(m)';

    % %
    % % STUDENT CODE GOES HERE: Compute Kernel PCA
    % %
    % Kernel PCA uses the spectral decomposition of the Gram matrix;
    % sort the eigenvalues and eigenvectors in descending order and
    % project the Gram matrix to "Lmax" dimensions
    
    % Calculate the singular values of Kmat
    [KEigVecs, KEigVals] = eig(Kmat, 'vector');
    
    %Order Eigenvectors such that vector associated with largest value is first
    [KEigVals, ndx] = sort(KEigVals, 'descend');
    KEigVecs = KEigVecs(:, ndx);
    
    %Find the scores for kernel PCA with Gram matrix
    Zgram = Kmat*KEigVecs(:, 1:Lmax);
    
    % Cluster the first two dimensions of the projection as 0,+1
    rng('default');
    cluster_idxs = kmeans(Zgram, 2) - 1;
    
   
    % %
    % % STUDENT CODE GOES HERE: plot and display results to console
    % %
    % Plot the labels and the clusters

    figure(5);
    gscatter(Zgram(:,1), Zgram(:,2), yvec, 'rb'),
    title('Kernel PCA Score Labelled by the Original Data Labels'),
    xlabel('Loading Vector 1'),
    ylabel('Loading Vector 2');

    figure(6);
    gscatter(Zgram(:,1), Zgram(:,2), cluster_idxs, 'mc'),
    title('Kernel PCA Score Labelled by the k-means Cluster Indices'),
    xlabel('Loading Vector 1'),
    ylabel('Loading Vector 2');
    
% END FUNCTION
end

function Kmat = gramgauss(Xmat, sigma2_in)
% K=GRAMGAUSS(X,SIGMA2)computes a Gram matrix for data in X
% using the Gaussian exponential exp(-1/sigma2*norm(X_i - X_j)^2)
%
% INPUTS:
%         X      - MxN data with M observations of N variables
%         sigma2 - optional scalar, default value is 1
% OUTPUTS:
%         K       NxN Gram matrix

    % Optionally use the provided sigma^2 scalar
    if (nargin>=2) && ~isempty('sigma2_in')
        sigma2 = sigma2_in;
    else
        sigma2 = 1;
    end
    
    % Finds distance between each observation in Xmat and divide by 
    % the hyperparameter "variance"
    Kmat = pdist2(Xmat, Xmat, 'squaredeuclidean') / sigma2;

    % Compute final Kmat / Gram matrix
    Kmat = exp(-Kmat);
   
end


% %
% % NO STUDENT CHANGES NEEDED BELOW HERE
% %
function waug = logreg(Xmat,yvec)
% WAUG=LOGREG(XMAT,YVEC) performs binary logistic regression on data
% matrix XMAT that has binary labels YVEC, using GLMFIT. The linear
% coefficients of the fit are in vector WAUG. Important note: the
% data XMAT are assumed to have no intercept term because these may be
% standardized data, but the logistic regression coefficients in WAUG
% will have an intercept term. The labels in YVEC are managed by
% >0 and ~>0, so either (-1,+1) convention or (0,1) convention in YVEC
% are acceptable.
%
% INPUTS:
%         XMAT - MxN array, of M observations in N variables
%         YVEC - Mx1 vector, binary labels
% OUTPUTS:
%         WAUG - (N+1)x1 vector, coefficients of logistic regression

    % Perform a circular shift of the GLMFIT coefficients so that
    % the final coefficient acts as an intercept term for XMAT
    
    warnstate = warning('query', 'last');
    warning('off');
    waug = circshift(glmfit(Xmat ,yvec>0, ...
        'binomial', 'link', 'probit'), -1);
    warning(warnstate);

    % END FUNCTION
end

function ph = plotline(vvec, color, lw, nv)
% PLOTLINE(VVEC,COLOR,LW,NV) plots a separating line
% into an existing figure
% INPUTS:
%        VVEC   - (M+1) augmented weight vector
%        COLOR  - character, color to use in the plot
%        LW   - optional scalar, line width for plotting symbols
%        NV   - optional logical, plot the normal vector
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. 

    % Set the line width
    if nargin >= 3 & ~isempty(lw)
        lwid = lw;
    else
        lwid = 2;
    end

    % Set the normal vector
    if nargin >= 4 & ~isempty(nv)
        do_normal = true;
    else
        do_normal = false;
    end

    % Current axis settings
    axin = axis();

    % Scale factor for the normal vector
    sval = 0.025*(axin(4) - axin(3));

    % Four corners of the current axis
    ll = [axin(1) ; axin(3)];
    lr = [axin(2) ; axin(3)];
    ul = [axin(1) ; axin(4)];
    ur = [axin(2) ; axin(4)];

    % Normal vector, direction vector, hyperplane scalar
    nlen = norm(vvec(1:2));
    uvec = vvec/nlen;
    nvec = uvec(1:2);
    dvec = [-uvec(2) ; uvec(1)];
    bval = uvec(3);

    % A point on the hyperplane
    pvec = -bval*nvec;

    % Projections of the axis corners on the separating line
    clist = dvec'*([ll lr ul ur] - pvec);
    cmin = min(clist);
    cmax = max(clist);

    % Start and end are outside the current plot axis, no problem
    pmin = pvec +cmin*dvec;
    pmax = pvec +cmax*dvec;

    % Create X and Y coordinates of a box for the current axis
    xbox = [axin(1) axin(2) axin(2) axin(1) axin(1)];
    ybox = [axin(3) axin(3) axin(4) axin(4) axin(3)];

    % Intersections of the line and the box
    [xi, yi] = polyxpoly([pmin(1) pmax(1)], [pmin(2) pmax(2)], xbox, ybox);

    % Point midway between the intersections
    pmid = [mean(xi) ; mean(yi)];

    % Range of the intersection line
    ilen = 0.5*norm([(max(xi) - min(xi)) ; (max(yi) - min(yi))]);

    % Plot the line according to the color specification
    hold on;
    if ischar(color)
        ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
            [color '-'], 'LineWidth', lwid);
    else
        ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
            'Color', color, 'LineStyle', '-', 'LineWidth', lwid);
    end
    if do_normal
        quiver(pmid(1), pmid(2), nvec(1)*ilen*sval, nvec(2)*ilen*sval, ...
            'Color', color, 'LineWidth', lwid, ...
            'MaxHeadSize', ilen/2, 'AutoScale', 'off');
    end
    hold off;
    
    % Remove this label from the legend, if any
    ch = get(gcf,'children');
    for ix=1:length(ch)
        if strcmp(ch(ix).Type, 'legend')
            ch(ix).String{end} = '';
        end
    end

% END FUNCTION
end