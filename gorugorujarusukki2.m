clc;clear;close all;
% --- 음원 로드 및 기본 설정 ---
[t_sound, fs] = audioread('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\siyouongen.wav');
load('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\simulated_rir\room_rir_8ch_0.1_-90_90_19.mat');
fs = double(fs);
target_SNR_dB = 0;
noise_ongen_ichi=[1];
fc = [500, 1000, 2000, 4000, 8000, 16000]; %lpf의 컷오프 주파수 ~fs/2까지
N = 200;    % 필터 차수 (짝수 권장, 높을수록 날카로움)
whatis_TandN=['w','b','p'];

start_time = fs * 10; % 10초부터
end_time = fs * 15;   % 15초까지 (5초 구간)
sample_length = end_time - start_time + 1;

% --- 2. 5초 길이에 맞게 노이즈 분할 생성 ---
% 전체 sample_length를 3구간으로 분할 (소수점 버림)
len1 = floor(sample_length / 3);
len2 = floor(sample_length / 3);
len3 = sample_length - len1 - len2; % 나머지 샘플까지 모두 포함

num_mics = size(mic_pos, 2);     % 8
mic_ch = num_mics;
num_sources = size(source_pos.', 2); % 19

saidaichi = cellfun(@length, rir);
pad_len = max(saidaichi(:));

%% =========================================================================
% --- [추가] 무한 증강형 딥러닝 학습 데이터셋 (Mixture) 생성 ---
% =========================================================================
fprintf('\n다중 각도 혼합 학습 데이터셋을 생성합니다...\n');

% 1. 파라미터 설정
num_train_samples = 20; % 만들고 싶은 믹스 데이터의 총 개수 (자유롭게 늘려도 됨)
interf_pool = [1:19]; % 타겟(10번)을 제외한 간섭 노이즈 각도 후보군

bandpassed_dataset=cell(mic_ch,num_train_samples);


% 2. 타겟 신호 8채널 매트릭스화 (기준 데이터)
% 길이는 이미 통일되어 있으므로 1번 채널 기준으로 잡음
len_sig = length(t_sound_rir_conv_source{1, 1});
target_8ch = zeros(len_sig, num_mics);
for ch = 1:num_mics
    target_8ch(:, ch) = t_sound_rir_conv_source{ch, 10}(:);
end
target_rms_noi = rms(t_sound_trimmed) / (10^(target_SNR_dB / 20));

directivity_target=cell(mic_ch,num_train_samples);

for k = 1:num_train_samples
    num_active_interf = randi([1, 3]); 
    % 간섭 노이즈 각도 랜덤하게 뽑기 (중복 없이)
    selected_angles = interf_pool(randperm(length(interf_pool), num_active_interf));
    
    TNN = whatis_TandN(randi(numel(whatis_TandN), length(selected_angles), 1));
    
    % 누적합을 위한 초기화 (8채널)
    for mic = 1:num_mics
        % fftfilt 후의 길이에 맞춰 초기화 (sample_length + pad_len - 1)
        directivity_target{mic, k} = zeros(sample_length + pad_len - 1, 1); 
    end
    
    for i = 1:length(TNN)
        color = TNN(i);
        
        % -5dB ~ +5dB 사이의 랜덤 SIR 스케일링 팩터 적용
        random_sir_dB = -5 + (10 * rand()); 
        scaling = 10^(-random_sir_dB / 20);
        
        % 노이즈 종류에 따른 생성
        switch color    
            case 'w'
                noise_matrix = single(randn(sample_length, 1)); 
            case 'p'
                noise_matrix = single(pinknoise(sample_length, 1));
            case 'b'
                brown = dsp.ColoredNoise('Color', 'brown', 'SamplesPerFrame', sample_length, 'NumChannels', 1);
                noise_matrix = single(brown());
        end
        
        % 공통 스케일링 및 패딩 적용
        noise_matrix = noise_matrix - mean(noise_matrix);
        noise_matrix = noise_matrix .* (target_rms_noi ./ rms(noise_matrix)) .* scaling; % scaling 추가
        noise__padded = [noise_matrix; zeros(pad_len, 1)];
        
        % 각 마이크 채널별 RIR 컨볼루션 및 누적
        for mic = 1:num_mics
            filtered_noise = fftfilt(rir{mic, selected_angles(i)}, noise__padded); % 인덱스를 mic로 수정
            directivity_target{mic, k} = directivity_target{mic, k} + filtered_noise(1:sample_length); % 노이즈 누적 혼합
        end
    end
end

for l = 1:num_train_samples
    for j = 1:mic_ch
        % ( ) 대신 { } 사용하여 셀 내부의 숫자 배열을 전달
        bpsignal = BPFilt_viaLPF(directivity_target{j, l}, fs, fc, lpf, N, 0); 
        bandpassed_dataset{j, l} = bpsignal;
    end
end

fprintf('총 %d개의 다중 혼합 학습 데이터셋 생성 완료!\n', num_train_samples);