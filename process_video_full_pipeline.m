clear; clc;

if count(py.sys.path, pwd) == 0
    insert(py.sys.path, int32(0), pwd);
end

py.importlib.import_module('unet_predict');
py.importlib.import_module('capsnet_predict');

fprintf('Loading U-Net...\n');
disp(char(py.unet_predict.load_model('unet_model_v2.h5')));

fprintf('Loading capsule network...\n');
disp(char(py.capsnet_predict.load_model('capsnet_v5.h5')));  % <-- adjust filename if yours differs

% Path to the standalone mediapipe environment's python.exe
mediapipeEnvPython = 'C:\Users\Ashika fathima\mediapipe_env\Scripts\python.exe';

%% CONFIG
inputVideoPath = 'dance_video.mp4';
outputVideoPath = 'dance_video_result.mp4';
frameSkip = 5;
tempFramePath = fullfile(pwd, 'temp_frame.jpg');

reader = VideoReader(inputVideoPath);
writer = VideoWriter(outputVideoPath, 'MPEG-4');
writer.FrameRate = reader.FrameRate;
open(writer);

frameCount = 0;
currentLabel = 'Detecting...';
currentConfidence = 0;

fprintf('Processing video: %s\n', inputVideoPath);

while hasFrame(reader)
    frame = readFrame(reader);
    frameCount = frameCount + 1;

    if mod(frameCount, frameSkip) == 1 || frameSkip == 1
        [h, w, ~] = size(frame);

        % --- Step 1: MediaPipe (external process) finds the hand ---
        imwrite(frame, tempFramePath);
        cmd = sprintf('"%s" detect_hand_standalone.py "%s"', mediapipeEnvPython, tempFramePath);
        [status, cmdout] = system(cmd);

        cmdout = strtrim(cmdout);
        outputLines = strsplit(cmdout, newline);
        outputLines = outputLines(~cellfun(@(s) isempty(strtrim(s)), outputLines));

        if isempty(outputLines)
            lastLine = '';
        else
            lastLine = strtrim(outputLines{end});
        end

        if status ~= 0 || strcmp(lastLine, 'NONE') || isempty(lastLine)
            currentLabel = 'No hand detected';
            currentConfidence = 0;
        else
            landmarksFlat = str2num(lastLine); %#ok<ST2NM>
            if numel(landmarksFlat) ~= 63
                currentLabel = 'No hand detected';
                currentConfidence = 0;
            else
                landmarks = reshape(landmarksFlat, 3, 21)';

                xs = landmarks(:,1) * w;
                ys = landmarks(:,2) * h;
                padX = (max(xs) - min(xs)) * 0.3;
                padY = (max(ys) - min(ys)) * 0.3;
                x1 = max(1, round(min(xs) - padX));
                y1 = max(1, round(min(ys) - padY));
                x2 = min(w, round(max(xs) + padX));
                y2 = min(h, round(max(ys) + padY));

                handCrop = frame(y1:y2, x1:x2, :);

                segmented = py.unet_predict.segment_for_capsnet(uint8(handCrop));
                segmentedMat = uint8(segmented);

                result = py.capsnet_predict.predict_capsnet(segmentedMat);
                currentLabel = char(result{1});
                currentConfidence = double(result{2});
            end
        end

        if mod(frameCount, frameSkip*10) == 1
            fprintf('Frame %d: %s (%.1f%%)\n', frameCount, currentLabel, currentConfidence*100);
        end
    end

    if currentConfidence > 0
        labelText = sprintf('%s (%.1f%%)', currentLabel, currentConfidence*100);
    else
        labelText = currentLabel;
    end

    annotatedFrame = insertText(frame, [20 20], labelText, ...
        'FontSize', 24, 'BoxColor', 'black', 'BoxOpacity', 0.6, 'TextColor', 'white');

    writeVideo(writer, annotatedFrame);
end

close(writer);
if exist(tempFramePath, 'file')
    delete(tempFramePath);
end
fprintf('Done. Annotated video saved to: %s\n', outputVideoPath);
