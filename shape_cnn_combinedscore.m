%% Clear environment
clear; clc; close all;

%% === Find the data
referenceFile = 'Y:\Shape_reconstruction\Data\Displacement_Kinovea_Data\References\ppt';
shapesFolder = 'Y:\Shape_reconstruction\Data\Displacement_Kinovea_Data\References\ppt';
shapes = dir(fullfile(shapesFolder, 'Shape_*'));
shapes_ordered = natsortfiles({shapes.name});   

path2SaveData = 'Y:\Shape_reconstruction\Data\Displacement_Kinovea_Data\References\ppt\Results\Score';
addpath('Y:\Matlab_scripts');

%% === Parameters
target_points = 200;
gridSize = 64;
rotationInvariant = true;

%% === Load and normalize reference shape
Reference_path = fullfile(referenceFile, 'Shape_0');
Tref = readtable(Reference_path);

if ~all(ismember({'x','y'}, Tref.Properties.VariableNames))
    error('Reference file missing x or y columns');
end

Reference_shape = [Tref.x, Tref.y];
valid_rows = all(~isnan(Reference_shape) & ~isinf(Reference_shape), 2);
Reference_shape_clean = Reference_shape(valid_rows, :);

Reference_shape_resampled = resampleShape2D(Reference_shape_clean, target_points);
centroid_ref = mean(Reference_shape_resampled);
Reference_shape_centered = Reference_shape_resampled - centroid_ref;

%% === Initialize result container
shape_results = struct([]);

%% === Define multi-scale CNN-inspired kernels
kernels = {
    fspecial('sobel'), ...                      % vertical edges
    fspecial('sobel')', ...                     % horizontal edges
    fspecial('prewitt'), ...                    % diagonal edge
    fspecial('prewitt')', ...                   % other diagonal
    fspecial('gaussian',[5 5],0.8), ...        % fine structure
    fspecial('gaussian',[9 9],1.5), ...        % coarse structure
    fspecial('log', [7 7], 1.0), ...           % Laplacian (blob detection)
};
numK = length(kernels);

%% === Process each shape
for i = 1:length(shapes_ordered)
    
    file_name = shapes_ordered{i};
    file_path = fullfile(shapesFolder, file_name);
    T = readtable(file_path);
    
    if ~all(ismember({'x','y'}, T.Properties.VariableNames))
        fprintf('⚠ Skipping file with missing columns: %s\n', file_name);
        continue;
    end
    shape = [T.x, T.y];
    valid_rows = all(~isnan(shape) & ~isinf(shape), 2);
    shape = shape(valid_rows, :);
    if isempty(shape)
        fprintf('⚠ Skipping invalid file: %s\n', file_name);
        continue;
    end
    
    % Resample and center
    shape_resampled = resampleShape2D(shape, target_points);
    centroid_shape = mean(shape_resampled);
    shape_centered = shape_resampled - centroid_shape;
    
    % Scale to match reference Frobenius norm
    scale_ref = norm(Reference_shape_centered, 'fro');
    scale_shape = norm(shape_centered, 'fro');
    shape_centered = shape_centered * (scale_ref / scale_shape);
    
    % Circular alignment
    [aligned_shape, ~] = circularAlign(Reference_shape_centered, shape_centered);
    
    % Optional rotation invariance
    if rotationInvariant
        best_score = -inf;
        best_rot_shape = aligned_shape;
        angles = -10:2:10;
        for theta = angles
            rotated_shape = rotateShape2D(aligned_shape, theta, [0,0]);
            refImg = rasterizeShape(Reference_shape_centered, gridSize);
            shapeImg = rasterizeShape(rotated_shape, gridSize);
            score = normalizedCrossCorrelation(refImg, shapeImg);
            if score > best_score
                best_score = score;
                best_rot_shape = rotated_shape;
            end
        end
        aligned_shape = best_rot_shape;
    end
    
    % Rasterize shapes
    refImg = rasterizeShape(Reference_shape_centered, gridSize);
    shapeImg = rasterizeShape(aligned_shape, gridSize);
    
    %% === Multi-metric scoring ===
    
    % 1. CNN Feature-based score
    ref_features = cell(numK,1);
    shape_features = cell(numK,1);
    feature_scores = zeros(numK,1);
    for k = 1:numK
        f_ref = conv2(refImg, kernels{k}, 'same');
        f_shape = conv2(shapeImg, kernels{k}, 'same');
        ref_features{k} = f_ref;
        shape_features{k} = f_shape;
        
        % Cosine similarity
        feature_scores(k) = dot(f_ref(:), f_shape(:)) / (norm(f_ref(:))*norm(f_shape(:))+eps);
    end
    CNN_score = mean(feature_scores);
    
    % 2. Direct pixel-based similarity (Dice coefficient)
    intersection = sum(refImg(:) & shapeImg(:));
    union = sum(refImg(:)) + sum(shapeImg(:));
    dice_score = 2*intersection / (union + eps);
    
    % 3. Structural Similarity Index (SSIM-like)
    ssim_score = ssim(refImg, shapeImg);
    
    % 4. Point-to-point distance (after alignment)
    point_distances = sqrt(sum((Reference_shape_centered - aligned_shape).^2, 2));
    mean_point_dist = mean(point_distances);
    max_point_dist = max(point_distances);
    rmse_points = sqrt(mean(point_distances.^2));
    
    % Normalize point distance to [0,1] score (0=bad, 1=perfect)
    % Using exponential decay: exp(-distance/scale)
    distance_scale = 0.1 * scale_ref; % 10% of reference size
    point_score = exp(-rmse_points / distance_scale);
    
    % 4b. Chamfer Distance (bidirectional)
    [chamfer_dist, chamfer_ref_to_shape, chamfer_shape_to_ref] = ...
        computeChamferDistance(Reference_shape_centered, aligned_shape);
    
    % Normalize Chamfer distance to [0,1] score
    chamfer_score = exp(-chamfer_dist / distance_scale);
    
    % 5. Perimeter ratio
    perimeter_ref = sum(sqrt(sum(diff(Reference_shape_centered).^2, 2)));
    perimeter_shape = sum(sqrt(sum(diff(aligned_shape).^2, 2)));
    perimeter_ratio = min(perimeter_ref, perimeter_shape) / max(perimeter_ref, perimeter_shape);
    
    % 6. Area ratio (using polygon area)
    area_ref = polyarea(Reference_shape_centered(:,1), Reference_shape_centered(:,2));
    area_shape = polyarea(aligned_shape(:,1), aligned_shape(:,2));
    area_ratio = min(area_ref, area_shape) / max(area_ref, area_shape);
    
    %% === Combined similarity score ===
    % Weighted average of different metrics
    weights = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05]; 
    % CNN, Dice, SSIM, Point, Chamfer, Perimeter, Area, Hausdorff
    scores_array = [CNN_score, dice_score, ssim_score, point_score, ...
                    chamfer_score, perimeter_ratio, area_ratio];
    combined_score = sum(weights(1:7) .* scores_array);
    
    % Combined difference map
    diffMap = zeros(size(refImg));
    for k = 1:numK
        diffMap = diffMap + abs(ref_features{k} - shape_features{k});
    end
    diffMap = diffMap ./ max(diffMap(:)+eps);
    
    %% === Visualization ===
    figure('Position', [100, 100, 1200, 400], 'Color', 'w');
    
    % Subplot 1: Overlaid shapes
    subplot(1,3,1);
    plot(Reference_shape_centered(:,1), Reference_shape_centered(:,2), 'b-', 'LineWidth', 2); hold on;
    plot(aligned_shape(:,1), aligned_shape(:,2), 'r--', 'LineWidth', 2);
    axis equal; grid on;
    legend('Reference', 'Test Shape', 'Location', 'best');
    title('Shape Overlay');
    
    % Subplot 2: Difference heatmap
    subplot(1,3,2);
    imagesc(diffMap); axis equal; axis off; colormap hot; colorbar;
    title('Feature Difference Map');
    
    % Subplot 3: Point-wise distances
    subplot(1,3,3);
    scatter(1:target_points, point_distances, 30, point_distances, 'filled');
    colormap(gca, 'jet'); colorbar;
    xlabel('Point Index'); ylabel('Distance from Reference');
    title('Point-to-Point Distances');
    grid on;
    
    sgtitle({file_name, ...
        sprintf('Combined: %.3f | CNN: %.3f | Dice: %.3f | SSIM: %.3f | Point: %.3f | Chamfer: %.3f', ...
        combined_score, CNN_score, dice_score, ssim_score, point_score, chamfer_score)}, ...
        'Interpreter', 'none');
    
    %% === Store results ===
    shape_results(i).file_name = file_name;
    shape_results(i).combined_score = combined_score;
    shape_results(i).CNN_score = CNN_score;
    shape_results(i).dice_score = dice_score;
    shape_results(i).ssim_score = ssim_score;
    shape_results(i).point_score = point_score;
    shape_results(i).chamfer_score = chamfer_score;
    shape_results(i).chamfer_distance = chamfer_dist;
    shape_results(i).chamfer_ref_to_shape = chamfer_ref_to_shape;
    shape_results(i).chamfer_shape_to_ref = chamfer_shape_to_ref;
    shape_results(i).mean_point_distance = mean_point_dist;
    shape_results(i).max_point_distance = max_point_dist;
    shape_results(i).rmse_points = rmse_points;
    shape_results(i).perimeter_ratio = perimeter_ratio;
    shape_results(i).area_ratio = area_ratio;
    shape_results(i).feature_scores = feature_scores(:)';
end

%% === Save results ===
if ~isempty(shape_results)
    results_table = struct2table(shape_results);
    writetable(results_table, fullfile(path2SaveData, 'Shape_Similarity_Scores.xlsx'));
    fprintf('✅ Excel file saved to: %s\n', fullfile(path2SaveData, 'Shape_Similarity_Scores.xlsx'));
    
    % Display summary statistics
    fprintf('\n=== Summary Statistics ===\n');
    fprintf('Mean Combined Score: %.3f (±%.3f)\n', mean([shape_results.combined_score]), std([shape_results.combined_score]));
    fprintf('Best Match: %s (Score: %.3f)\n', shape_results(1).file_name, max([shape_results.combined_score]));
    [~, best_idx] = max([shape_results.combined_score]);
    fprintf('Best Match: %s (Score: %.3f)\n', shape_results(best_idx).file_name, shape_results(best_idx).combined_score);
end

%% === Helper Functions ===
function resampled = resampleShape2D(shape, numPoints)
    diffs = diff(shape);
    keep = [true; any(diffs,2)];
    shape = shape(keep,:);
    x = shape(:,1); y = shape(:,2);
    d = [0; cumsum(sqrt(diff(x).^2 + diff(y).^2))];
    [d, uniqueIdx] = unique(d);
    x = x(uniqueIdx); y = y(uniqueIdx);
    if length(d) < 2
        resampled = repmat(shape(1,:), numPoints, 1);
        return;
    end
    xi = linspace(0,d(end),numPoints);
    resampled = [interp1(d,x,xi)', interp1(d,y,xi)'];
end

function img = rasterizeShape(shape, gridSize)
    if isempty(shape)
        img = zeros(gridSize);
        return;
    end
    minX = min(shape(:,1)); maxX = max(shape(:,1));
    minY = min(shape(:,2)); maxY = max(shape(:,2));
    rangeX = maxX - minX; rangeY = maxY - minY;
    if rangeX < eps, rangeX = 1; end
    if rangeY < eps, rangeY = 1; end
    
    x = (shape(:,1)-minX)/rangeX*(gridSize-1)+1;
    y = (shape(:,2)-minY)/rangeY*(gridSize-1)+1;
    x = min(max(round(x),1),gridSize);
    y = min(max(round(y),1),gridSize);
    
    img = zeros(gridSize);
    for j = 1:length(x)
        img(y(j), x(j)) = 1;
    end
    img = imgaussfilt(img, 1.5);
end

function [alignedShape, bestShift] = circularAlign(refShape, shapeToAlign)
    N = size(refShape,1);
    minRMSE = inf; bestShift = 0; alignedShape = shapeToAlign;
    for shift = 0:N-1
        shifted = circshift(shapeToAlign, shift, 1);
        rmse = sqrt(sum((refShape - shifted).^2, 'all')/numel(refShape));
        if rmse < minRMSE
            minRMSE = rmse;
            bestShift = shift;
            alignedShape = shifted;
        end
    end
end

function rotatedShape = rotateShape2D(shape, angleDeg, center)
    theta = deg2rad(angleDeg);
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    rotatedShape = (shape - center) * R' + center;
end

function score = normalizedCrossCorrelation(A,B)
    A = A - mean(A(:)); B = B - mean(B(:));
    score = sum(A(:).*B(:))/sqrt(sum(A(:).^2)*sum(B(:).^2)+eps);
end

function [chamfer_dist, dist_ref_to_shape, dist_shape_to_ref] = computeChamferDistance(shape1, shape2)
    % Compute bidirectional Chamfer distance between two shapes
    % Chamfer distance is the average of:
    %   1. Mean minimum distance from each point in shape1 to shape2
    %   2. Mean minimum distance from each point in shape2 to shape1
    %
    % This is more robust than point-to-point distance because it doesn't
    % assume point correspondence
    
    % For each point in shape1, find closest point in shape2
    dist_ref_to_shape = zeros(size(shape1, 1), 1);
    for i = 1:size(shape1, 1)
        dists = sqrt(sum((shape2 - shape1(i,:)).^2, 2));
        dist_ref_to_shape(i) = min(dists);
    end
    
    % For each point in shape2, find closest point in shape1
    dist_shape_to_ref = zeros(size(shape2, 1), 1);
    for i = 1:size(shape2, 1)
        dists = sqrt(sum((shape1 - shape2(i,:)).^2, 2));
        dist_shape_to_ref(i) = min(dists);
    end
    
    % Chamfer distance is the mean of both directions
    chamfer_dist = (mean(dist_ref_to_shape) + mean(dist_shape_to_ref)) / 2;
    
    % Alternative: you can use RMS instead of mean for different behavior
    % chamfer_dist = sqrt((mean(dist_ref_to_shape.^2) + mean(dist_shape_to_ref.^2)) / 2);
end