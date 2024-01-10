function a3_20273229
% Function for CISC271, Winter 2022, Assignment #3

    % %
    % % STUDENT CODE GOES HERE: REMOVE THIS COMMENT
    % % Read the test data from a CSV file
    % % Extract the data matrix and the Y cluster identifiers
    % % Compute the pair of columns of Xmat with the lowest DB index
    % % Compute the PCA's of the data using the SVD; score the clusterings
    % % Display the cluster scores and the indexes; plot the data
    % %
    data_table = readtable("wine.csv", "ReadRowNames", true);
    data_mat = data_table{:,:};
    data_mat = transpose(data_mat);
    label_vec = data_mat(:,1);
    data_mat(:,1) = [];
    zeromean_mat = data_mat - mean(data_mat);
    [n_rows, n_cols] = size(zeromean_mat);
    
    num_clusters = 3; % For 3 different cultivars

    % Figures
    f1 = figure;
    f2 = figure;
    f3 = figure;

    % Part 1: Finding the best pair
    % db_idx will be a 13 x 3 matrix where the first column contains the
    % DB indices and the second and third col contain the column indicies 
    db_idxs = []; 
    for i = 1:n_cols
        for j = (i+1):n_cols
            X = data_mat(:, [i, j]);
            db_idxs(end + 1, :) = [dbindex(X, label_vec), i, j];
        end
    end
    % Contains lowest lowest DB index value and the row index
    [lowest_db, best_idx] = min(db_idxs(:,1));
    % Contains the column indices for the pair of varaibles
    best_vals = [db_idxs(best_idx, 2), db_idxs(best_idx, 3)]
    parta_db = lowest_db
    figure(f1);
    gscatter(zeromean_mat(:,best_vals(1)), zeromean_mat(:,best_vals(2)), label_vec),
    xlabel('Dimension 1 (Varaible 1)'), ylabel('Dimension 2 (Variable 7)'), 
    title('Dimensionality Reduction Using Data Columns with Lowest DB Index');

    % Optional code for k-means
    % [cluster_idxs, centroids] = kmeans([zeromean_mat(:,best_vals(1)), zeromean_mat(:,best_vals(2))], num_clusters);
    % gscatter(zeromean_mat(:,best_vals(1)), zeromean_mat(:,best_vals(2)), cluster_idxs);
    % clustera_db = dbindex([zeromean_mat(:,best_vals(1)), zeromean_mat(:,best_vals(2))], cluster_idxs)
    

    % Part 2: Reduce 13D to 2D
    [U, S, V] = svd(zeromean_mat);
    reduced_mat1 = zeromean_mat * [V(:,1), V(:,2)];
    % Scoring reduction using the first two principle components
    [reduced_idx1, reduced_centroids1] = kmeans(reduced_mat1, num_clusters);
    partb_db = dbindex(reduced_mat1, label_vec)
    figure(f2);
    gscatter(reduced_mat1(:,1), reduced_mat1(:,2), label_vec),
    xlabel('Dimension 1 (First Principle Component)'), ylabel('Dimension 2 (Second Principle Component)'), 
    title('Dimensionality Reduction Using PCA on the Zero-mean Data Matrix');

    % Optional code for k-means
    % gscatter(reduced_mat1(:,1), reduced_mat1(:,2), reduced_idx1);
    % clusterb_db = dbindex(reduced_mat1, reduced_idx1)

    % Part 3: Scoring the reduction of standardized data
    standard_mat = zscore(data_mat);
    [U, S, V] = svd(standard_mat);
    % Scoring reduction using the first two principle components
    reduced_mat2 = standard_mat * [V(:, 1), V(:, 2)];
    [reduced_idx2, reduced_centroids2] = kmeans(standard_mat, num_clusters);
    partc_db = dbindex(reduced_mat2, label_vec)
    figure(f3);
    gscatter(reduced_mat2(:,1), reduced_mat2(:,2), label_vec),
    xlabel('Dimension 1 (First Principle Component)'), ylabel('Dimension 2 (Second Principle Component)'), 
    title('Dimensionality Reduction Using PCA on the Standardized Data Matrix');
    
    % Optional code for k-means
    % gscatter(reduced_mat2(:,1), reduced_mat2(:,2), reduced_idx2);
    % clusterc_db = dbindex(reduced_mat2, reduced_idx2)
end
function score = dbindex(Xmat, lvec)
% SCORE=DBINDEX(XMAT,LVEC) computes the Davies-Bouldin index
% for a design matrix XMAT by using the values in LVEC as labels.
% The calculation implements a formula in their journal article.
%
% INPUTS:
%        XMAT  - MxN design matrix, each row is an observation and
%                each column is a variable
%        LVEC  - Mx1 label vector, each entry is an observation label
% OUTPUT:
%        SCORE - non-negative scalar, smaller is "better" separation

    % Anonymous function for Euclidean norm of observations
    rownorm = @(xmat) sqrt(sum(xmat.^2, 2));

    % Problem: unique labels and how many there are
    kset = unique(lvec);
    k = length(kset);

    % Loop over all indexes and accumulate the DB score of each cluster
    % gi is the cluster centroid
    % mi is the mean distance from the centroid
    % Di contains the distance ratios between IX and each other cluster
    D = [];
    for ix = 1:k
        Xi = Xmat(lvec==kset(ix), :);
        gi = mean(Xi);
        mi = mean(rownorm(Xi - gi));
        Di = [];
        for jx = 1:k
            if jx~=ix
                Xj = Xmat(lvec==kset(jx), :);
                gj = mean(Xj);
                mj = mean(rownorm(Xj - gj));
                Di(end+1) = (mi + mj)/norm(gi - gj);
            end
        end
        D(end+1) = max(Di);
    end

    % DB score is the mean of the scores of the clusters
    score = mean(D);
end
