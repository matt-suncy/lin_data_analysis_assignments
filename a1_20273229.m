function set12 = a1_20273229(elist) %replace 00000000 with yor student number and rename file
% Function for CISC271, Winter 2022, Assignment #1
%
% IN:
%     elist - Mx2 array of edges, each row is a pair of vertices
% OUT:
%     set12 - Nx1 vertex clsutering, -1 for SET1 and +1 for SET2
    
    %% Finding the Fiedler Vector
    % Problem size: number of vertices in the graph
    n = max(elist(:));
    

    % %
    % % 20273229: replace this trivial adjacency matrix
    % %     with your computation; the first line is a hint for how to
    % %     initialize your matrix correctly
    % %

    % Creates a matrix of all zeros
    A = zeros(n);
    
    % Iterate through the list and set exisiting edges to 1
    k = 10
    for i = 1:size(elist)
        a = elist(i, 1);
        b = elist(i, 2);
        if (a ~= k) && (b ~= k)
            A(a, b) = 1;
            A(b, a) = 1;
        end
    end

    % Calculates the diagonal matrix whose diagonal entries are the sum of
    % each row of A
    DVec = sum(A, 2);
    D = diag(DVec);
    % Formula for finding the Laplacian matrix
    L = D - A;
    
    % Retrieving the the eigenvectors of the Laplacian matrix
    [eigVec1, eigVal1] = eig(L);
    % The Fiedler vec we want is the second eigenvector
    fiedlerVec = eigVec1(:,2);

    %% Changing the Fielder Vector to contain 1 and -1s
    % %
    % % 20273229: replace this constant clustering vector
    % %     with your computation that uses the Fiedler vector; the
    % %     vector SET12 should be plus/minus 1 for automated grading
    %
    
    % If the value in the Fiedler vec is >= 0 it becomes 1 and otherwise it
    % is 0
    set12 = fiedlerVec >= 0;
    % 1s stays as 1s and 0s become -1s
    set12 = (set12 * 2) - 1;

    % %
    % % 20273229: replace this trivial display to console
    % %     with your computation from the vector SET12
    % %

    disp('Set 1 vertices are:');
    % Non-negative values are categorized into set 1
    disp(set12 >= 0)
    disp('Set 2 vertices are:');
    % Negative values are categorized into set 2
    disp(set12 < 0)


    % Plot the graph, Cartesian and clustered
    plot271a1(A, set12);
end

function plot271a1(Amat, cvec)
% PLOTCLUSTER(AMAT,CVEC) plots the adjacency matrix AMAT twice;
% first, as a Cartesian grid, and seconnd, by using binary clusters
% in CVEC to plot the graph of AMAT based on two circles
%
% INPUTS: 
%         Amat - NxN adjacency matrix, symmetric with binary entries
%         cvec - Nx1 vector of class labels, having 2 distinct values
% OUTPUTS:
%         none
% SIDE EFFECTS:
%         Plots into the current figure

    % %
    % % Part 1 of 2: plot the graph as a rectangle
    % %

    % Problem size
    [m n] = size(Amat);

    % Factor the size into primes and use the largest as the X size
    nfact = factor(n);
    nx = nfact(end);
    ny = round(n/nx);

    % Create a grid and pull apart into coordinates; offset Y by +2
    [gx, gy] = meshgrid((1:nx) - round(nx/2), (1:ny) + 2);

    % Offset the odd rows to diagram the connections a little better
    for ix=1:2:ny
        gx(ix, :) = gx(ix, :) + 0.25*ix;
    end

    % The plot function needs simple vectors to create the graph
    x = gx(:);
    y = flipud(gy(:));

    % Plot the graph of A using the Cartesian grid
    plot(graph(tril(Amat, -1), 'lower'), 'XData', x, 'YData', y);
    axis('equal');

    % %
    % % Part 2 of 2: plot the graph as pair of circles
    % %
    % Set up the X and Y coordinates of each graph vertex
    xy = zeros(2, numel(cvec));

    % Number of cluster to process
    kset = unique(cvec);
    nk = numel(kset);

    % Base circle is radius 2, points are centers of clusters
    bxy = 2*circlen(nk);

    % Process each cluster
    for ix = 1:nk
        jx = cvec==kset(ix);
        ni = sum(jx);
        xy(:, jx) = bxy(:, ix) + circlen(ni);
    end

    hold on;
    plot(graph(Amat), 'XData', xy(1,:), 'YData', xy(2,:));
    hold off;
    title(sprintf('Clusters of (%d,%d) nodes', ...
        sum(cvec==kset(1)), sum(cvec==kset(2))));
end

function xy = circlen(n)
% XY=CIRCLEN(N) finds N 2D points on a unit circle
%s
% INPUTS:
%         N  - positive integer, number of points
% OUTPUTS:
%         XY - 2xN array, each column is a 2D point

    xy = [cos(2*pi*(0:(n-1))/n) ; sin(2*pi*(0:(n-1))/n)];
end
