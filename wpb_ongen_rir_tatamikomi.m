clc;clear;close all;
% --- 음원 로드 및 기본 설정 ---
[t_sound, fs] = audioread('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\siyouongen.wav');
load('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\simulated_rir\room_rir_8ch_0.1_-90_90_19.mat');
fs = double(fs);
target_SNR_dB = 0;
noise_ongen_ichi=[1];
fc = [500, 1000, 2000, 4000, 8000, 16000]; %lpf의 컷오프 주파수 ~fs/2까지
N = 200;    % 필터 차수 (짝수 권장, 높을수록 날카로움)


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

% --- 1. 타깃 음원 처리 (단일 음원) ---
t_sound_mono = mean(t_sound, 2); 
t_sound_trimmed = t_sound_mono(start_time:end_time, :); 

target_rms_noi = rms(t_sound_trimmed) / (10^(target_SNR_dB / 20));

% 1. 화이트, 핑크, 브라운 노이즈 각각 생성
noise_matrix_white = single(randn(len1, num_sources)); 
noise_matrix_pink = single(pinknoise(len2, num_sources));
brown = dsp.ColoredNoise('Color', 'brown', 'SamplesPerFrame', len3, 'NumChannels', num_sources);
noise_matrix_brown = single(brown());

% 2. DC 오프셋 제거 (각 열의 평균을 0으로)
noise_matrix_white = noise_matrix_white - mean(noise_matrix_white);
noise_matrix_pink = noise_matrix_pink - mean(noise_matrix_pink);
noise_matrix_brown = noise_matrix_brown - mean(noise_matrix_brown);

% 3. 개별 스케일링 (for 문 없이 행렬 연산으로 한 번에 처리)
noise_matrix_white = noise_matrix_white .* (target_rms_noi ./ rms(noise_matrix_white));
noise_matrix_pink = noise_matrix_pink .* (target_rms_noi ./ rms(noise_matrix_pink));
noise_matrix_brown = noise_matrix_brown .* (target_rms_noi ./ rms(noise_matrix_brown));

% 4. 최종 병합
noise_matrix = [noise_matrix_white; noise_matrix_pink; noise_matrix_brown];

% --- 3. 각 앵글별 SNR 스케일링 (For 문 사용) ---
target_rms_noi = rms(t_sound_trimmed) / (10^(target_SNR_dB / 20));
scaled_noise_matrix = zeros(size(noise_matrix));

for j = 1:num_sources
    current_noise = noise_matrix(:, j);
    % 각 열(각도)의 노이즈 RMS를 타깃 RMS에 맞게 개별 스케일링
    scaled_noise_matrix(:, j) = current_noise * (target_rms_noi / rms(current_noise));
end

% 첫 번째 각도 기준 SNR 확인 출력
final_snr_check = 20 * log10(rms(t_sound_trimmed) / rms(scaled_noise_matrix(:, 1)));
fprintf('최종 확인된 SNR (1번 각도 기준): %.4f dB\n', final_snr_check);

% --- 4. 패딩 (Padding) ---
saidaichi = cellfun(@length, rir);
pad_len = max(saidaichi(:));

t_sound_padded = [t_sound_trimmed; zeros(pad_len, 1)];
noise_padded_matrix = [scaled_noise_matrix; zeros(pad_len, num_sources)];

% --- 5. RIR 컨볼루션 (각도 매핑) ---
s_o_r = size(rir); % [8, 19]
centered_0_ongen = (s_o_r(2) + 1) / 2; % 10번 음원(0도) 중심

t_sound_rir_conv_source = cell(s_o_r(1), 1);
noise_rir_conv_source = cell(s_o_r);

% 타깃 신호는 항상 0도(10번 인덱스)에서 온다고 가정
for i = 1:num_mics
    t_sound_rir_conv_source{i, 1} = fftfilt(rir{i, centered_0_ongen}, t_sound_padded);
end

% 🚨 노이즈 신호는 각 RIR 각도(j)에 맞춰 독립된 노이즈 매트릭스의 j번째 열을 사용
for j = 1:num_sources
    for i = 1:num_mics
        noise_rir_conv_source{i, j} = fftfilt(rir{i, j}, noise_padded_matrix(:, j));
    end
end

% --- 6. 메모리 최적화형 클리핑 방지 노말리제이션 (cell2mat 제거) ---
% 각 셀에서 최댓값을 바로 추출하여 RAM 폭발 방지
max_noise_vals = cellfun(@(x) max(abs(x)), noise_rir_conv_source);
max_target_vals = cellfun(@(x) max(abs(x)), t_sound_rir_conv_source);

max_noise = max(max_noise_vals(:));
max_target = max(max_target_vals(:));
overall_max = max(max_noise, max_target);

scale_factor = 1 / overall_max;

% 스케일 적용
noise_rir_conv_source = cellfun(@(x) x * scale_factor, noise_rir_conv_source, 'UniformOutput', false);
t_sound_rir_conv_source = cellfun(@(x) x * scale_factor, t_sound_rir_conv_source, 'UniformOutput', false);

% 불필요한 거대 변수 즉시 삭제
clear noise_matrix scaled_noise_matrix t_sound_padded noise_padded_matrix max_noise_vals max_target_vals;
noise_mat=cell2mat(noise_rir_conv_source);
target_mat=cell2mat(t_sound_rir_conv_source);

max_noise = max(abs(noise_mat), [], 'all');
max_target = max(abs(target_mat), [], 'all');

overall_max = max(max_noise, max_target)
%%
audio_to_play = double(t_sound_rir_conv_source{1,1});
% sound(audio_to_play,single(fs));
%%
%음성합성
for i = 1:num_mics
    % 혹시 모를 가로/세로 벡터 오류를 막기 위해 (:) 사용 (무조건 열벡터로)
    sig_n = noise_rir_conv_source{i, noise_ongen_ichi(1,1)}(:);%1번 음원채널의 노이즈(모노)
    sig_t = t_sound_rir_conv_source{i, 1}(:);%10번음원채널의 음성(모노)
    
    minimi = min(length(sig_n), length(sig_t));
    gouseionseibako{i, 1} = sig_n(1:minimi) + sig_t(1:minimi);
end



%%
%서브밴드처리=> 밴드패스된 7ch cell데이터를 8x19행렬로
    %% 1. 파라미터 설정
     % 샘플링 주파수 (예시)
    % 나카지마 논문 기반 예시 주파수 (6개 경계 -> 7개 대역)
    inputSignal=cell2mat(t_sound_rir_conv_source');
    
    
    %% 2. LPF 설계 (fir1 사용)
    % 모든 필터는 동일한 차수 N을 가져야 위상 지연이 일치하여 PR이 가능함
lpf = cell(1, length(fc));
for i = 1:length(fc)
    lpf{i} = fir1(N, fc(i)/(fs/2));
end
    
directivity_target=noise_rir_conv_source;%각도별로 재생할 음원

bandpassed_dataset=cell(mic_ch,num_sources);%8x19, 각 셀에는 1x7의 밴드패스된 셀이 격납
for i=1:num_sources
    for j=1:mic_ch
        bpsignal=BPFilt_viaLPF(directivity_target(j,i),fs,fc,lpf,N,0);
        bandpassed_dataset{j,i}=bpsignal;%{}로 안넣으면 셀에 격납 안됨
    end
end

% for문으로 다시합쳐서 퍼펙트리컨스트럭트 확인