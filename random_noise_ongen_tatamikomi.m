clc; clear; close all;
version = 16;
N = 1.5; % 지향성 계수 (변수명 N에서 변경)
anycomment = 'sdr+0.1rmse+softfilt+clipping+동적계수';

% =========================================================================
% 1. 기본 설정 및 음원/RIR 로드
% =========================================================================
[t_sound, fs] = audioread('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\siyouongen.wav');
load('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\simulated_rir\room_rir_8ch_0.1_-90_90_19.mat');
fs = double(fs);



% 파라미터 설정
target_SNR_dB = 0;
fc = [500, 1000, 2000, 4000, 8000, 16000]; % LPF 컷오프
filter_order = 200; % 필터 차수 (변수명 N에서 변경)
whatis_TandN = ['w', 'b', 'p'];
num_train_samples = 200; 
interf_pool = [1:9,11:19];
center_mic = 5; 




%% 2. LPF 설계 (fir1 사용)
    % 모든 필터는 동일한 차수 N을 가져야 위상 지연이 일치하여 PR이 가능함
lpf = cell(1, length(fc));
for i = 1:length(fc)
    lpf{i} = fir1(filter_order, fc(i)/(fs/2));
end
%% 
start_time = fs * 10; % 10초부터
end_time = fs * 12;   % 15초까지 (5초 구간)
sample_length = end_time - start_time + 1;

num_mics = size(mic_pos, 2);     % 8채널
mic_ch = num_mics;
num_sources = size(source_pos.', 2); % 19개 각도
deg_in = -90:10:90; % 10도 간격 스캔 각도

saidaichi = cellfun(@length, rir);
pad_len = max(saidaichi(:));
target_rms_noi = rms(t_sound(start_time:end_time)) / (10^(target_SNR_dB / 20));

% =========================================================================
% 2. 다중 각도 혼합 신호(X) 및 정답 신호(Y) 동시 생성
% =========================================================================
fprintf('\n다중 각도 혼합 학습 데이터셋(X)과 정답(Y)을 생성합니다...\n');


directivity_target = cell(mic_ch, num_train_samples); % X (혼합 입력)
Y_target_raw = cell(1, num_train_samples);            % Y (정답 타겟)
brown = dsp.ColoredNoise('Color', 'brown', 'SamplesPerFrame', sample_length, 'NumChannels', 1);%브라운 객체는 한번만 불러도됨

for k = 1:num_train_samples
    num_active_interf = randi([1, 3]); 
    random_interf=interf_pool(randperm(length(interf_pool), num_active_interf));
    selected_angles = [10, random_interf];

    TNN = whatis_TandN(randi(numel(whatis_TandN), length(selected_angles), 1));
    
    % 누적합을 위한 초기화
    for mic = 1:num_mics
        directivity_target{mic, k} = zeros(sample_length, 1); 
    end
    Y_target_raw{1, k} = zeros(sample_length, 1);
    
    for i = 1:length(TNN)
        color = TNN(i);
        current_angle_idx = selected_angles(i);
        
        if current_angle_idx == 10
            scaling = 1.0; 
        else
            random_sir_dB = -5 + (10 * rand()); 
            scaling = 10^(-random_sir_dB / 20);
        end

        switch color    
            case 'w'
                noise_matrix = randn(sample_length, 1); 
            case 'p'
                noise_matrix = pinknoise(sample_length, 1);
            case 'b'
                noise_matrix = brown();
        end
        
        noise_matrix = noise_matrix - mean(noise_matrix);
        noise_matrix = noise_matrix .* (target_rms_noi ./ rms(noise_matrix)) .* scaling;
        noise__padded = [noise_matrix; zeros(pad_len - 1, 1)];
        
        % 현재 섞이는 각도의 수식 Weight 계산
        weight = (1/N)^(abs(deg_in(current_angle_idx))/10);
        
        for mic = 1:num_mics
            filtered_noise = fftfilt(rir{mic, current_angle_idx}, noise__padded);
            filtered_trimmed = filtered_noise(1:sample_length); % 길이 맞춤
            
            % X: 모든 채널 누적 혼합
            directivity_target{mic, k} = directivity_target{mic, k} + filtered_trimmed;
            
            % Y: 센터 마이크에 대해서만 Weight 적용하여 정답지에 누적
            if mic == center_mic
                Y_target_raw{1, k} = Y_target_raw{1, k} + (filtered_trimmed * weight);
            end
        end
    end
end

% =========================================================================
% 3. BPF (Bandpass Filter) 적용
% =========================================================================
num_bands = length(fc) + 1;
bandpassed_X = cell(mic_ch, num_train_samples);
bandpassed_Y = cell(1, num_train_samples);

for l = 1:num_train_samples
    % X 서브밴드 필터링 (8채널)
    for j = 1:mic_ch
        bpsignal_X = BPFilt_viaLPF(directivity_target(j, l),fs,fc,lpf, filter_order, 0); 
        bandpassed_X{j, l} = bpsignal_X;
    end
    % Y 서브밴드 필터링 (1채널)
    bpsignal_Y = BPFilt_viaLPF(Y_target_raw(1, l), fs,fc,lpf, filter_order, 0);
    bandpassed_Y{1, l} = bpsignal_Y;
end
fprintf('다중 혼합 데이터 및 타겟 데이터 필터링 완료!\n');







% =========================================================================
% 4. 딥러닝 데이터 구성 및 학습 (밴드별 독립 모델)
% =========================================================================
bc = 2048 * 4;
maxEpochs = 500;
inputSize = num_mics;      
outputSize = 256;  
numResponses = 1;   
VF = 10/5;
VP = 50/5;

networks = cell(num_bands, 1);
global_max = cell(num_bands, 1);

for band_idx = 1:num_bands
    fprintf('==== %d번 밴드 네트워크 데이터 구성 및 학습 ====\n', band_idx);
    chunkSize = bc; 
    
    X_train_concat = []; 
    Y_train_concat = []; 
    
    % num_train_samples(혼합된 데이터들)를 루프로 돌며 행렬 조립
    for k = 1:num_train_samples
        % X 행렬 조립 [sample_length x 8]
        ch_matrix = zeros(sample_length, mic_ch);
        for ch_idx = 1:mic_ch
            ch_matrix(:, ch_idx) = bandpassed_X{ch_idx, k}{band_idx}(:);
        end
        
        % Y 행렬 조립 [sample_length x 1]
        y_matrix = bandpassed_Y{1, k}{band_idx}(:);
        
        % 미니배치 사이즈에 맞게 클리핑
        numChunks = floor(sample_length / chunkSize);
        X_train_concat = [X_train_concat; ch_matrix(1:numChunks*chunkSize, :)];
        Y_train_concat = [Y_train_concat; y_matrix(1:numChunks*chunkSize, :)];
    end
    
    % 정규화 상수 계산 (0 나누기 방지)
    band_max = sqrt(mean(X_train_concat.^2, 'all')) + 10^-8;
    if band_max == 0; band_max = 1; end 
    
    % 훈련/검증 분할 (8:2 셔플)
    numChunksTotal = floor(size(X_train_concat, 1) / chunkSize);
    rng(42); 
    shuffledChunkIdx = randperm(numChunksTotal);
    
    valStartChunk = floor(numChunksTotal * 0.8) + 1;
    train_chunks = shuffledChunkIdx(1:valStartChunk-1);
    val_chunks = shuffledChunkIdx(valStartChunk:end);
    
    X_train_new = zeros(length(train_chunks) * chunkSize, inputSize);
    Y_train_new = zeros(length(train_chunks) * chunkSize, 1);
    X_val_new = zeros(length(val_chunks) * chunkSize, inputSize);
    Y_val_new = zeros(length(val_chunks) * chunkSize, 1);
    
    for i = 1:length(train_chunks)
        src_start = (train_chunks(i) - 1) * chunkSize + 1;
        src_end   = train_chunks(i) * chunkSize;
        dst_start = (i - 1) * chunkSize + 1;
        dst_end   = i * chunkSize;
        
        X_train_new(dst_start:dst_end, :) = X_train_concat(src_start:src_end, :);
        Y_train_new(dst_start:dst_end, :) = Y_train_concat(src_start:src_end, :);
    end
    
    for i = 1:length(val_chunks)
        src_start = (val_chunks(i) - 1) * chunkSize + 1;
        src_end   = val_chunks(i) * chunkSize;
        dst_start = (i - 1) * chunkSize + 1;
        dst_end   = i * chunkSize;
        
        X_val_new(dst_start:dst_end, :) = X_train_concat(src_start:src_end, :);
        Y_val_new(dst_start:dst_end, :) = Y_train_concat(src_start:src_end, :);
    end
    
    % 최종 정규화
    X_train_final = X_train_new / band_max;
    Y_train_final = Y_train_new / band_max;
    X_val = X_val_new / band_max;
    Y_val = Y_val_new / band_max;

    % 네트워크 구성 및 학습
    layers = [
        featureInputLayer(inputSize)
        fullyConnectedLayer(outputSize)
        leakyReluLayer(0.01)
        fullyConnectedLayer(outputSize/2)
        leakyReluLayer(0.01)
        fullyConnectedLayer(numResponses, Bias = 0, BiasLearnRateFactor = 0)
    ];
    
    options = trainingOptions("adam", ... 
        InitialLearnRate = 0.001, ... 
        MaxEpochs = maxEpochs, ...
        miniBatchSize = bc, ...    
        GradientThreshold = Inf, ...
        Plots = "training-progress", ...
        Shuffle = 'never', ...
        ExecutionEnvironment = "auto", ...
        ValidationData = {X_val, Y_val}, ... 
        ValidationFrequency = floor((size(X_train_final, 1)/bc)/VF), ... 
        ValidationPatience = VP,  ...          
        OutputNetwork = 'best-validation-loss', ... 
        Verbose = 0);
    
    networks{band_idx} = trainnet(X_train_final, Y_train_final, layers, @sisdrLossLayer, options);
    global_max{band_idx, 1} = band_max;
end

disp('✅ 7개 밴드 네트워크 학습 모두 완료!');
file_name = [anycomment, 'customizedLoss_NNBF_learningdata_subandsoft_N', num2str(N), 'version_', num2str(version), '.mat'];
save(file_name, 'networks', 'global_max');