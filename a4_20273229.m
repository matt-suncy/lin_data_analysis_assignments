% TODO
% Put confusion matrix calculations into function
% Change accuracy calculations for measuring optimal threshold

function a4_20273229
% Function for CISC271, Winter 2022, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);

    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 17;
    jOB = 16;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM zDM qOB zOB] = a4q1(rDM, yDM, rOB, yOB);

    % %
    % % STUDENT CODE GOES HERE: PLOT RELEVANT DATA
    % %

    % Figures
    f1 = figure;
    f2 = figure;
    f3 = figure;
    f4 = figure;

    figure(f1);
    gscatter(rDM(:,1), rDM(:,2), yDM),
    xlabel('First Loading Vector'), ylabel('Second Loading Vector'), 
    title('Dimensionality Reduction Using PCA for the Diabetes Label with Binary Labels');

    figure(f2);
    gscatter(rOB(:,1), rOB(:,2), yOB),
    xlabel('First Loading Vector'), ylabel('Second Loading Vector'), 
    title('Dimensionality Reduction Using PCA for the Obesity Label with Binary Labels');

    figure(f3);
    gscatter(zDM, ones(size(rDM, 1), 1).*yDM, yDM),
    xlabel('LDA Scores'), title('Visualization of LDA Scores for the Diabetes Label with Binary Labels');
    axis([-5, 5, -1.5, 1.5]);
    disp('LDA axis for diabetes lablel:')
    disp(qDM)
    
    figure(f4);
    gscatter(zOB, ones(size(rOB, 1), 1).*yOB, yOB),
    xlabel('LDA Scores'), title('Visualization of LDA Scores for the Obesity Label with Binary Labels');
    axis([-5, 5, -1.5, 1.5]);
    disp('LDA axis for obesity lablel:')
    disp(qOB)

    % Compute the ROC curve and its AUC where: "xroc" is the horizontal
    % axis of false positive rates; "yroc" is the vertical
    % axis of true positive rates; "auc" is the area under curve
    % %
    % % STUDENT CODE GOES HERE: COMPUTE, PLOT, DISPLAY RELEVANT DATA
    % %

    [fprDM, tprDM, aucDM, boptDM] = roccurve(yDM, zDM);
    [fprOB, tprOB, aucOB, boptOB] = roccurve(yOB, zOB);
    
    f5 = figure;
    figure(f5);
    plot(fprDM, tprDM),
    xlabel('False Positive Rate'), ylabel('True Positive Rate'),
    text(0.6, 0.4, 'AUC = 0.9309', 'FontSize', 14),
    title('ROC Curve for the Diabetes Label');
    disp('Area under curve for diabetes:')
    disp(aucDM)
    disp('Optimal threshold for diabetes:')
    disp(boptDM)
    disp('Corresponding optimal confusion matrix:')
    disp(confmat(yDM, zDM, boptDM))

    f6 = figure;
    figure(f6);
    plot(fprOB, tprOB),
    xlabel('False Positive Rate'), ylabel('True Positive Rate'),
    text(0.6, 0.4, 'AUC = 0.6071', 'FontSize', 14),
    title('ROC Curve for the Obesity Label');
    disp('Area under curve for obesity:')
    disp(aucOB)
    disp('Optimal threshold for obesity:')
    disp(boptOB)
    disp('Corresponding optimal confusion matrix:')
    disp(confmat(yOB, zOB, boptOB))


% END OF FUNCTION
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % %
    % % STUDENT CODE GOES HERE: COMPUTE SCORES USING LDA AXES
    % %
    Xmean1 = mean(Xmat1);
    z1 = (Xmat1 - ones(size(Xmean1, 1), 1)*Xmean1)*q1;

    Xmean2 = mean(Xmat2);
    z2 = (Xmat2 - ones(size(Xmean2, 1), 1)*Xmean2)*q2;
    
    
% END OF FUNCTION
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation

    qvec = ones(size(X1,2), 1);
    xbar1 = mean(X1);
    xbar2 = mean(X2);
    

    % Compute the within-class means and scatter matrices
    % %
    % % STUDENT CODE GOES HERE: COMPUTE S1, S2, Sw
    % %

    % Form M1 and M2
    M1 = X1 - ones(size(X1, 1), 1)*xbar1;
    M2 = X2 - ones(size(X2, 1), 1)*xbar2;

    % Computing within-label scatter matrices
    S1 = M1' * M1;
    S2 = M2' * M2;
    Sw = S1 + S2;

    
    % Compute the between-class scatter matrix
    % %
    % % STUDENT CODE GOES HERE: COMPUTE Sb
    % %
    
    % Computing between label scatter matrix
    xbar = [xbar1 ; xbar2];
    Sb = (xbar - mean(xbar))' * (xbar - mean(xbar));

    % Fisher's linear discriminant is the largest eigenvector
    % of the Rayleigh quotient
    % %
    % % STUDENT CODE GOES HERE: COMPUTE qvec
    % %

    % Calculate LDA axis as vMAX of Sw^(-1)*Sb
    [eigvecs, eigvals] = eig(Sw \ Sb);
    qvec = eigvecs(:,1);

    % May need to correct the sign of qvec to point towards mean of X1
    if (xbar1 - xbar2)*qvec < 0
        qvec = -qvec;
    end
% END OF FUNCTION
end

function [fpr, tpr, auc, bopt] = roccurve(yvec_in,zvec_in)
% [FPR TPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
% ROC curve and related values for labels YVEC and scores ZVEC.
% Unique scores are used as thresholds for binary classification.
%
% INPUTS:
%         YVEC - Mx1 labels, +/- computed as ==/~= 1
%         ZVEC - Mx1 scores, real numbers
% OUTPUTS:
%         FPR  - Kx1 vector of False Positive Rate values
%         TPR  - Kx1 vector of  True Positive Rate values
%         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
%         BOPT - scalar, optimal threshold for accuracy

    % Sort the scores and permute the labels accordingly
    [zvec, zndx] = sort(zvec_in);
    yvec = yvec_in(zndx);
        
    % Sort and find a unique subset of the scores; problem size
    bvec = unique(zvec);
    bm = numel(bvec);
    
    % Compute a confusion matrix for each unique threshold value;
    % extract normalized entries into TPR and FPR vectors; track
    % the accuracy and optimal B threshold
    tpr = [];
    fpr = [];
    acc = -inf;
    bopt = -inf;
    for jx = 1:bm
        % %
        % % STUDENT CODE GOES HERE: FIND TPR, FPR, OPTIMAL THRESHOLD
        % %
        
        % Threshold the predictions
        B = bvec(jx);

        cm = confmat(yvec,zvec,B);
        % Compute TPR and FPR
        tp = cm(1,1);
        fn = cm(1,2);
        fp= cm(2,1);
        tn = cm(2,2);
        tpr(jx) = tp / (tp + fn);
        fpr(jx) = fp / (fp + tn);

        % Compare accuracy to find optimal threshold
        acc_jx = (tpr(jx) * (tn / (tn + fp))) - (fpr(jx) * (fn / (fn + tp)));
        if acc_jx > acc
            acc = acc_jx;
            bopt = B;
        end
    end

    % Ensure that the rates, from these scores, will plot correctly
    tpr = sort(tpr);
    fpr = sort(fpr);
    auc = aucofroc(fpr, tpr);
end
    
function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

    % Find the plus/minus 1 vector of quantizations
    qvec = sign((zvec >= theta) - 0.5);
    
    % Compute the confusion matrix by entries
    % %
    % % STUDENT CODE GOES HERE: COMPUTE MATRIX
    % %

    % Compute the confusion matrix
    tp = sum(yvec == 1 & qvec == 1);
    fp = sum(yvec == -1 & qvec == 1);
    tn = sum(yvec == -1 & qvec == -1);
    fn = sum(yvec == 1 & qvec == -1);
    cmat = [tp, fn; fp, tn];

end

function auc = aucofroc(fpr, tpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X, undx] = sort(reshape(fpr, 1, numel(fpr)));
    Y = sort(reshape(tpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 X 1] , [0 Y 1]));
end
