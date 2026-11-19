clear; clc;

dataFolder = fullfile(pwd, 'train');
imagesFolder = fullfile(pwd, 'unet_images_v2');
masksFolder = fullfile(pwd, 'unet_masks_v2');
imagesPerClass = 150;
imgSize = [128 128];
numBackgroundVariants = 3;  % how many different backgrounds per image

if ~exist(imagesFolder, 'dir'), mkdir(imagesFolder); end
if ~exist(masksFolder, 'dir'), mkdir(masksFolder); end

backgrounds = generateSyntheticBackgrounds(imgSize(1), imgSize(2), 10);

classFolders = dir(dataFolder);
classFolders = classFolders([classFolders.isdir] & ~startsWith({classFolders.name}, '.'));

counter = 0;
for c = 1:numel(classFolders)
    className = classFolders(c).name;
    srcDir = fullfile(dataFolder, className);
    imgFiles = [dir(fullfile(srcDir, '*.jpg')); dir(fullfile(srcDir, '*.png')); dir(fullfile(srcDir, '*.jpeg'))];
    numToUse = min(imagesPerClass, numel(imgFiles));
    fprintf('Generating masks for %s: %d images x %d backgrounds\n', className, numToUse, numBackgroundVariants);

    for k = 1:numToUse
        img = imread(fullfile(imgFiles(k).folder, imgFiles(k).name));
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end

        mask = segmentHand(img);  % accurate mask from the real green screen
        imgR = imresize(img, imgSize);
        maskR = imresize(mask, imgSize, 'nearest');
        handOnlyR = imgR;
        for ch = 1:3
            channel = handOnlyR(:,:,ch);
            channel(~maskR) = 0;
            handOnlyR(:,:,ch) = channel;
        end

        labelImg = uint8(maskR) + 1;  % 1=background, 2=hand - SAME for every variant

        for v = 1:numBackgroundVariants
            bg = backgrounds{randi(numel(backgrounds))};
            composite = bg;
            mask3 = repmat(maskR, [1 1 3]);
            composite(mask3) = handOnlyR(mask3);

            counter = counter + 1;
            fname = sprintf('img_%05d.png', counter);
            imwrite(composite, fullfile(imagesFolder, fname));
            imwrite(labelImg, fullfile(masksFolder, fname));
        end
    end
end

fprintf('Done. %d image/mask pairs generated (with background variety).\n', counter);

function mask = segmentHand(img)
    hsv = rgb2hsv(img);
    h = hsv(:,:,1); s = hsv(:,:,2); v = hsv(:,:,3);
    isGreen = (h > 0.22 & h < 0.45) & (s > 0.25) & (v > 0.2);
    mask = ~isGreen;
    mask = bwareaopen(mask, 300);
    mask = imfill(mask, 'holes');
    mask = imopen(mask, strel('disk', 3));

    cc = bwconncomp(mask);
    if cc.NumObjects > 1
        sizes = cellfun(@numel, cc.PixelIdxList);
        [~, biggest] = max(sizes);
        cleanMask = false(size(mask));
        cleanMask(cc.PixelIdxList{biggest}) = true;
        mask = cleanMask;
    end
end

function backgrounds = generateSyntheticBackgrounds(h, w, n)
    backgrounds = cell(1, n);
    for i = 1:n
        kind = mod(i, 4);
        switch kind
            case 0  % solid black (fixes your exact failure case)
                backgrounds{i} = zeros(h, w, 3, 'uint8');
            case 1  % solid random color
                color = uint8(randi([0 255], 1, 1, 3));
                backgrounds{i} = repmat(color, [h w 1]);
            case 2  % gradient
                grad = uint8(linspace(0, 255, w));
                imgg = repmat(grad, [h 1]);
                backgrounds{i} = cat(3, imgg, flip(imgg,2), uint8(128*ones(h,w)));
            case 3  % noise texture
                backgrounds{i} = uint8(randi([0 255], h, w, 3));
        end
    end
end
