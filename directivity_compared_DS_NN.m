
centered_mic_num=5;
directivity_target=noise_rir_conv_source;%각도별로 재생할 음원



load('NNBF_learningdata.mat')%rt60_noise
% 2. 테스트용 섞인 신호 준비 [N x 8]
% test_signal = cell2mat(t_sound_rir_conv_source(:,1)') ; 
test_signal = cell2mat(gouseionseibako(:,1)') ; 

% 3. 🚨 불러온 data_max로 정규화 수행
test_input_norm = test_signal / global_max;

% 4. 빔포밍 수행
output_norm = predict(net, test_input_norm);

% 5. 원래 볼륨으로 복원
clean_audio = output_norm * global_max;
max(clean_audio)

% clean_audio=bandpass(clean_audio,[300,3000],fs);
% 6. 오디오 파일로 저장

% soundsc(clean_audio, fs);
% soundsc(saishuonsei,fs);

%%
% 19개 각도 정의
angles_deg = linspace(-90, 90, num_sources); 
output_energy = zeros(1, num_sources);

% [중요] 루프 진입 전 주파수 빈(Bin) 개수 사전 확정
% 임의로 1번째 채널 신호 길이를 파악하여 크기를 지정합니다.
DS_sample_len = length(directivity_target{1, 1}); 
num_freq_bins = floor(DS_sample_len/2) + 1;

% 1번부터 19번 각도까지 루프 구동
for angle_idx = 1:num_sources
    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} = directivity_target{i,angle_idx}(:);
    end
    
    % 네트워크를 통한 각도별 신호 추정 및 스케일 복원
    steered_signal = mean(cell2mat(dna_scan),2);%Nx1
    
    % --- 루프 안에서 FFT 수행 ---
    nnsig_length = length(steered_signal);
    fft_nnsig = fft(steered_signal);
    P2 = abs(fft_nnsig / nnsig_length);
    P1 = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    % [핵심] 현재 각도(행) 자리에 FFT 스펙트럼(열)을 가로로 대입
    DS_Gain_Matrix(angle_idx, :) = P1(:)'; 
    
    % 기존 RMS 에너지 계산 유지
    output_energy(angle_idx) = rms(steered_signal);
end

% --- 루프가 끝난 후 3D 플로팅 데이터 준비 ---
f = fs * (0:(num_freq_bins-1)) / DS_sample_len; % 주파수 축 배열 (1 x num_freq_bins)

% [논문 표준] 주파수 게인을 dB 스케일로 변환 및 최고점을 0dB로 정규화
% 리니어 스케일로 그리면 감쇄 대역(Null)이 평평하게 묻혀서 안 보입니다.
DS_Gain_Matrix_dB = 20 * log10(DS_Gain_Matrix / max(DS_Gain_Matrix(:)));

figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);

% 3D가 아닌 2D 이미지 형태로 출력 (위에서 내려다봄)
imagesc(angles_deg, f, DS_Gain_Matrix_dB'); 

% y축(주파수)이 아래에서 위로 올라가도록 방향 뒤집기 (imagesc의 기본 특성 보정)
set(gca, 'YDir', 'normal'); 

% 색상은 기본 parula(최신 매트랩 논문 표준) 또는 gray 사용
colormap parula; % 또는 colormap gray;
colorbar;
caxis([-80 0]); % -30dB 이하는 뭉뚱그림

xlabel('Angle (Degrees)', 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontWeight', 'bold');
title('DS Spatial Frequency Beampattern', 'FontWeight', 'bold');
ylim([0 fs/2]); % 타겟 주파수 대역


%%
% 화이트노이즈에 대한 지향성 플롯

% =========================================================================
% --- 3. 주파수 대역 및 각도별 3D 지향성 패턴 (Beampattern) 분석 ---
% =========================================================================
disp('3D 공간 주파수 지향성 패턴을 계산 중입니다...');

% [중요] 루프 진입 전 주파수 빈(Bin) 개수 사전 확정
% 임의로 1번째 채널 신호 길이를 파악하여 크기를 지정합니다.
NN_sample_len = length(directivity_target{1, 1}); 
num_freq_bins = floor(NN_sample_len/2) + 1;

% 3D 데이터를 담을 2차원 행렬 사전 할당 (행: 각도수, 열: 주파수빈수)
NN_Gain_Matrix = zeros(num_sources, num_freq_bins);

% 1번부터 19번 각도까지 루프 구동
for angle_idx = 1:num_sources
    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} = directivity_target{i,angle_idx}(:)/global_max;
    end
    
    % 네트워크를 통한 각도별 신호 추정 및 스케일 복원
    steered_signal = predict(net, cell2mat(dna_scan));
    steered_signal = global_max * steered_signal;
    
    % --- 루프 안에서 FFT 수행 ---
    nnsig_length = length(steered_signal);
    fft_nnsig = fft(steered_signal);
    P2 = abs(fft_nnsig / nnsig_length);
    P1 = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    % [핵심] 현재 각도(행) 자리에 FFT 스펙트럼(열)을 가로로 대입
    NN_Gain_Matrix(angle_idx, :) = P1(:)'; 
    
    % 기존 RMS 에너지 계산 유지
    output_energy(angle_idx) = rms(steered_signal);
end

% --- 루프가 끝난 후 3D 플로팅 데이터 준비 ---
f = fs * (0:(num_freq_bins-1)) / NN_sample_len; % 주파수 축 배열 (1 x num_freq_bins)

% [논문 표준] 주파수 게인을 dB 스케일로 변환 및 최고점을 0dB로 정규화
% 리니어 스케일로 그리면 감쇄 대역(Null)이 평평하게 묻혀서 안 보입니다.
NN_Gain_Matrix_dB = 20 * log10(NN_Gain_Matrix / max(NN_Gain_Matrix(:)));

figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);

% 3D가 아닌 2D 이미지 형태로 출력 (위에서 내려다봄)
imagesc(angles_deg, f, NN_Gain_Matrix_dB'); 

% y축(주파수)이 아래에서 위로 올라가도록 방향 뒤집기 (imagesc의 기본 특성 보정)
set(gca, 'YDir', 'normal'); 

% 색상은 기본 parula(최신 매트랩 논문 표준) 또는 gray 사용
colormap parula; % 또는 colormap gray;
colorbar;
caxis([-80 0]); % -30dB 이하는 뭉뚱그림

xlabel('Angle (Degrees)', 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontWeight', 'bold');
title('DNNBF Spatial Frequency Beampattern', 'FontWeight', 'bold');
ylim([0 fs/2]); % 타겟 주파수 대역

fprintf("DSBF max dB: %5.4f DSBF max dB: %5.4f \n",max(DS_Gain_Matrix_dB(:)),min(DS_Gain_Matrix_dB(:)));
fprintf("NNBF max dB: %5.4f NNBF max dB: %5.4f \n",max(NN_Gain_Matrix_dB(:)),min(NN_Gain_Matrix_dB(:)));


%%
%SI-SDR など 要修正
s=cell2mat(directivity_target(centered_mic_num,1));

L=length(s);
DS_s_hat=saishuonsei(1+shifts(centered_mic_num):L+shifts(centered_mic_num),:);%센터마이크 기준으로 딜레이 설정
NN_s_hat=predict(net,test_input_norm)*global_max;
interfer=noise_rir_conv_source{centered_mic_num,noise_ongen_ichi(1,1)};


[DS_SI_SDR, DS_SI_SIR, DS_SI_SAR, DS_e_targetpow, DS_e_interfpow, DS_e_artifpow]=SI_metrics(s,DS_s_hat,interfer);%SI_SDR, SI_SIR, SI_SAR, e_targetpow, e_interfpow, e_artifpow
[NN_SI_SDR, NN_SI_SIR, NN_SI_SAR, NN_e_targetpow, NN_e_interfpow, NN_e_artifpow]=SI_metrics(s,NN_s_hat,interfer);%SI_SDR, SI_SIR, SI_SAR, e_targetpow, e_interfpow, e_artifpow


fprintf('\n======================================================\n');
fprintf('        [ %s ] 빔포밍 성능 분석 결과 \n', datestr(now, 'HH:MM:SS'));
fprintf('======================================================\n');
fprintf(' Metric |   DAS (Baseline)   |   NN (Proposed)    \n');
fprintf('------------------------------------------------------\n');
fprintf(' SI-SDR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SDR, NN_SI_SDR);
fprintf(' SI-SIR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SIR, NN_SI_SIR);
fprintf(' SI-SAR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SAR, NN_SI_SAR);
fprintf('------------------------------------------------------\n');
fprintf(' Target |    %10.2e      |    %10.2e      \n', DS_e_targetpow, NN_e_targetpow);
fprintf(' Interf |    %10.2e      |    %10.2e      \n', DS_e_interfpow, NN_e_interfpow);
fprintf(' Artif  |    %10.2e      |    %10.2e      \n', DS_e_artifpow, NN_e_artifpow);
fprintf('======================================================\n');

nnsig_length=length(steered_signal);



