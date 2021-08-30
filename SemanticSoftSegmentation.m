
% Semantic Soft Segmentation
% This function implements the soft segmentation approach described in
% Yagiz Aksoy, Tae-Hyun Oh, Sylvain Paris, Marc Pollefeys, Wojciech Matusik
% "Semantic Soft Segmentation", ACM TOG (Proc. SIGGRAPH) 2018

function [softSegments, initSoftSegments, Laplacian, affinities, features, superpixels, eigenvectors, eigenvalues] = SemanticSoftSegmentation(image, features)
    
    disp('Semantic Soft Segmentation')
    % Prepare the inputs and superpixels
    image = im2double(image);
    if size(features, 3) > 3 % If the features are raw, hyperdimensional, preprocess them
        features = preprocessFeatures(features, image);
    else
        features = im2double(features);
    end
    superpixels = Superpixels(image);
    [h, w, ~] = size(image);

    disp('     Computing affinities')
    % Compute the affinities and the Laplacian
    affinities{1} = mattingAffinity(image);
    affinity = affinities{1};

    affinities{2} = superpixels.neighborAffinities(features); % semantic affinity
    affinity = affinities{2};

    affinities{3} = superpixels.nearbyAffinities(image); % non-local color affinity (set to 0 since it is not color images)
    Laplacian = affinityMatrixToLaplacian(0.3 * affinities{1} + 0.7 * affinities{2} + 0 * affinities{3}); % Equation 6 color 0.01
%%%%% 0.9 + 0.1 for astro %%%%%% tunable parameters
    disp('     Computing eigenvectors')
    % Compute the eigendecomposition
    eigCnt = 30; % We use 100 eigenvectors in the optimization
    [eigenvectors, eigenvalues] = eigs(Laplacian, eigCnt, 'SM');

    
    disp('     Initial optimization')
    
    % Compute initial soft segments
    initialSegmCnt =30;%30 %% tunable parameter
    sparsityParam = 1; %0.3 %% tunable parameter
    iterCnt = 40; %40 %% tunable but not found to makes significant difference
    % feeding features to the function below triggers semantic intialization
    initSoftSegments = softSegmentsFromEigs(eigenvectors, eigenvalues, Laplacian, ...
                                            h, w, features, initialSegmCnt, iterCnt, sparsityParam, [], []);

    % Group segments w.r.t. their mean semantic feature vectors
    groupedSegments = groupSegments(initSoftSegments, features, 8); %%%%%%% default 6 for astro %%%% tunable parameters
    %%% 

    
    disp('     Final optimization')
    % Do the final sparsification
    softSegments = sparsifySegments(groupedSegments, Laplacian, imageGradient(image, false, 5));
    
    disp('     Done.')
end