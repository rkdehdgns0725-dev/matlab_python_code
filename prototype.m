clc;clear;close all;


[noise,noise_fs]=audioread('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\white_gaussian_noise.wav');
[t_sound,fs]=audioread('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\siyouongen.wav');
load('C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\simulated_rir\room_rir_8ch_0.5_-90_90_19.mat');
fs=double(fs);
target_SNR_dB = 12;
c=343;%m/s


room_rt=rt60(cell2mat(rir(1,10)),fs)

start_time=fs*10;%10초부터
end_time=fs*30;%30초까지

t_sound_mono = mean(t_sound, 2); 
noise_mono = mean(noise, 2);
t_sound=t_sound_mono(start_time:end_time,:);
noise=noise_mono(start_time:end_time,:);

source_pos_t = source_pos.'; 
num_mics = size(mic_pos, 2);     % 8
mic_ch=num_mics;
num_sources = size(source_pos_t, 2); %c 19
gouseionseibako = cell(num_mics, 1);
co_micarray = mean(mic_pos, 2); % 마이크 어레이 중심 (3 x 1)
noise_ongen_ichi=[1];


%%


s_o_r=size(rir);%[8,19] 마이크 8채널 / 음원 19개위치

centered_0_ongen=(s_o_r(2)+1)/2; %10번 음원이 중심

noise_rir_conv_source=cell(s_o_r); %rir이 적용된 noise신호
t_sound_rir_conv_source=cell(s_o_r(1),1);%rir이 적용된 target_sound 신호

%%
% scaling
noise=noise/(rms(noise)/rms(t_sound));
start_snr=20*log10(rms(t_sound)/rms(noise));
fprintf('音源とのSNR : %.4f dB \n',start_snr)

% SNR = 20 * log10(rms_sig / target_rms_noi)
target_rms_noi = rms(t_sound) / (10^(target_SNR_dB / 20));
scaled_noise = noise * (target_rms_noi / rms(noise));

final_snr = 20 * log10(rms(t_sound) / rms(scaled_noise));
fprintf('최종 확인된 SNR: %.4f dB\n', final_snr);

%% padding length

% saidaichi=[];
% for i=1:length(rir(:,1))
%     for j=1:length(rir(1,:))
%         saidaichi=[saidaichi,length(rir{i,j})];
%     end
% end
%%동일코드
saidaichi = cellfun(@length, rir);%cellfun(length(rir)) 즉 rir행렬안의 요소의 길이를 
saidaichi = saidaichi(:)';
pad_len=max(saidaichi)
%%
% rir이 적용된 target and noise sound//
% conv을 zp된 원신호(target, noise)와 fftfilt를 이용//원리->fft 후 ifft//
%circular conv

t_sound_padded = [t_sound; zeros(pad_len, 1)];
noise_padded=[scaled_noise;zeros(pad_len,1)];
for i=1:length(rir(:,1))
    t_sound_rir_conv_source{i,1}=fftfilt(rir{i,centered_0_ongen},t_sound_padded);%음원10번채널 모노음성
end

for i=1:length(rir(:,1))
    for j=1:length(rir(1,:))
        t_sound_rir_conv_source_8_18{i,j}=fftfilt(rir{i,j},t_sound_padded);
    end
end


for i=1:length(rir(:,1))
    for j=1:length(rir(1,:))
        noise_rir_conv_source{i,j}=fftfilt(rir{i,j},noise_padded);
    end
end
%%
%클리핑 방지 최종 노말리제이션/

noise_mat=cell2mat(noise_rir_conv_source);
target_mat=cell2mat(t_sound_rir_conv_source);

max_noise = max(abs(noise_mat), [], 'all');
max_target = max(abs(target_mat), [], 'all');

overall_max = max(max_noise, max_target)
scale_factor=0.9/overall_max;


% scale_factor는 아까 구한 단 하나의 숫자(상수)라고 가정합니다.
% noise_rir_conv_source의 모든 셀에 scale_factor를 곱함
noise_rir_conv_source = cellfun(@(x) x * scale_factor, noise_rir_conv_source, 'UniformOutput', false);

% t_sound_rir_conv_source의 모든 셀에도 동일하게 적용
t_sound_rir_conv_source = cellfun(@(x) x * scale_factor, t_sound_rir_conv_source, 'UniformOutput', false);

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

