function [rmsvars lowndx rmstrain rmstest] = a2_20273229
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the file "GOODS.CSV" by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that is
% best explained by the other variables, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    filename = 'goods.csv';
    [rmsvars lowndx] = a2q1(filename);
    [rmstrain rmstest] = a2q2(filename, lowndx)

end

function [rmsvars, lowndx] = a2q1(filename)
% [RMSVARS LOWNDX]=A2Q1(FILENAME) finds the RMS errors of
% linear regression of the data in the file FILENAME by treating
% each column as a vector of dependent observations, using the other
% columns of the data as observations of independent varaibles. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. 
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWNDX   - integer scalar, index into RMSVALS

    % Read the test data from a CSV file; find the size of the data
    goods_data = readtable(filename,'ReadRowNames',true);
    Amat = goods_data{:,:};
    A_size = size(Amat);
    % Number of columns in Amat
    m = A_size(2);

    % Compute the RMS errors for linear regression of each column vec
    Xmat = zscore(Amat);
    rmsvars = zeros(1,m);
    for i = 1:m
        cvec = Xmat(:,i);
        A = Xmat;
        A(:,i) = [];
        % Essentially equivalent to using normal equation, there exists
        % some subtle differences
        wvec = A \ cvec;
        pvec = A * wvec;
        rmsval = rms(cvec - pvec);
        % Storing RMS error in row vector
        rmsvars(i) = rmsval;
    end
    % Finding the index of the min RMS error and saving it
    [min_val, min_index] = min(rmsvars);
    lowndx = min_index;

    % Find the regression on your choice of standardized
    % or unstandardized variables
    % Choice was made to plot the understandardized regression and data
    A = Amat;
    c = Amat(:,lowndx);
    A(:,lowndx) = [];
    pvec = A * (A \ c);

    % Plot the results
    hold on;
    plot(pvec,'-o',Color='b'), xlabel('Time (3 Month Intervals)'), ylabel('Price (USD)'), 
    title('Regression of Variable with Smallest RMS error: Copper');
    plot(c,'o', Color='r');
    hold off;

end
function [rmstrain, rmstest] = a2q2(filename,lowndx)
% [RMSTRAIN RMSTEST]=A3Q2(LOWNDX) finds the RMS errors of 5-fold
% cross-validation for the variable LOWNDX of the data in the file
% FILENAME. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         FILENAME - character string, name of file to be processed;
%                    assume that the first row describes the data variables
%         LOWNDX   - integer scalar, index into the data
% OUTPUTS:
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    % Read the test data from a CSV file; find the size of the data
    goods_data = readtable(filename,'ReadRowNames',true);
    Amat = goods_data{:,:};
    [nrows, ncols] = size(Amat);

    % Create Xmat and yvec from the data and the input parameter,
    % accounting for no standardization of data
    % Leave Xmat undstardized, this will seem redundant but this way it is
    % eaiser to standardize data later if desired
    Xmat = zscore(Amat);
    yvec = Xmat(:,lowndx);
    Xmat(:,lowndx) = [];
    
    % Compute the RMS errors of 5-fold cross-validation
    [rmstrain, rmstest] = mykfold(Xmat, yvec, 5);
end

function [rmstrain,rmstest]=mykfold(Xmat, yvec, k_in)
% [RMSTRAIN,RMSTEST]=MYKFOLD(XMAT,yvec,K) performs a k-fold validation
% of the least-squares linear fit of yvec to XMAT. If K is omitted,
% the default is 5.
%
% INPUTS:
%         XMAT     - MxN data vector
%         yvec     - Mx1 data vector
%         K        - positive integer, number of folds to use
% OUTPUTS:
%         RMSTRAIN - 1xK vector of RMS error of the training fits
%         RMSTEST  - 1xK vector of RMS error of the testing fits

    % Problem size
    M = size(Xmat, 1);

    % Set the number of folds; must be 1<k<M
    if nargin >= 3 & ~isempty(k_in)
        k = max(min(round(k_in), M-1), 2);
    else
        k = 5;
    end

    % Initialize the return variables
    rmstrain = zeros(1, k);
    rmstest  = zeros(1, k);

    rng('default') % makes sure get same resuts each time (consistent seed)
    randidx = randperm(linspace(1,size(Xmat,1),1)); % random permutation of rows
    num_in_fold = round(size(Xmat,1)/k);

    % Process each fold
    for ix=1:k
        Xperm = Xmat(randidx, :);
        yperm = yvec(randidx, :);
        
        % Calculating start and end of test indices depedning on the value
        % of k
        current_test_start = (ix - 1) * num_in_fold + 1;
        current_test_end = current_test_start + num_in_fold - 1;
        % To make sure index will not exceed number of rows 
        if current_test_end > M
            current_test_end = M;
        end

        test_indices = false(size(Xmat,1), 1); % create logical array of 0s
        test_indices(current_test_start:current_test_end) = true; % set test indices to true (extract)
        train_indices = true(size(Xmat,1), 1); % create logical array of 1s
        train_indices(current_test_start:current_test_end) = 0; % set test indices to false (remove test extract train)
        
        % extract data for train and test
        xmat_test = Xperm(test_indices,:);
        yvec_test = yperm(test_indices,:);
        xmat_train = Xperm(train_indices,:);
        yvec_train = yperm(train_indices,:);
           
        % trained model weight vector
        wvec = xmat_train \ yvec_train;
           
        % rms errors of training and testing set
        rmstrain(ix) = rms(xmat_train*wvec - yvec_train);
        rmstest(ix)  = rms(xmat_test*wvec  - yvec_test);
    end

end
