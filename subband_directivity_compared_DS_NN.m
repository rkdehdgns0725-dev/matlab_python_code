
centered_mic_num=5;
directivity_target=noise_rir_conv_source;%각도별로 재생할 음원
% directivity_target=t_sound_rir_conv_source_8_18;%각도별로 재생할 음원



load('NNBF_learningdata_v2.mat')%rt60_noise
% 2. 테스트용 섞인 신호 준비 [N x 8]
test_signal = t_sound_rir_conv_source(:,1) ; 
% test_signal = cell2mat(gouseionseibako(:,1)') ; 
% test_signal = gouseionseibako(:,1) ; 

bpsig_to_net=BPFilt_viaLPF(test_signal,fs,fc,lpf,N,1);

tl=length(bpsig_to_net{1,1});%신호길이

subband_saishu_onsei=zeros(tl,num_bands);%서브밴드 처리된 음성저장용
for band_idx=1:num_bands %지금 상태에선 2~4번밴드만 합쳤을 때 가장 성능이 좋은거같다?
    %  🚨 불러온 data_max로 정규화 수행
    test_input_norm=bpsig_to_net{1,band_idx}/ global_max{band_idx};
    max(test_input_norm,[],"all")
    subband_saishu_onsei(:,band_idx)=predict(networks{band_idx},test_input_norm)*global_max{band_idx};

end
clean_audio=sum(subband_saishu_onsei,2);

% 5. 원래 볼륨으로 복원
max(clean_audio)

% clean_audio=bandpass(clean_audio,[300,3000],fs);
% 6. 오디오 파일로 저장

soundsc(clean_audio, fs);
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
% =========================================================================
% --- 3. 주파수 대역 및 각도별 3D 지향성 패턴 (Beampattern) 분석 ---
% =========================================================================
disp('3D 공간 주파수 지향성 패턴을 계산 중입니다...');

NN_sample_len = length(directivity_target{1, 1}); 
num_freq_bins = floor(NN_sample_len/2) + 1;

% [수정] 전체 대역용 행렬과 밴드별 행렬을 독립적으로 분리하여 초기화
NN_Gain_Matrix = zeros(num_sources, num_freq_bins); 
BPF_NN_Gain_Matrix = cell(1, num_bands);
for b = 1:num_bands
    BPF_NN_Gain_Matrix{b} = zeros(num_sources, num_freq_bins); % 각 셀에 독립된 행렬 배치
end

% 1번부터 19번 각도까지 루프 구동
for angle_idx = 1:num_sources
    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} = directivity_target{i,angle_idx}(:);
    end
    
    dna_scan = BPFilt_viaLPF(dna_scan', fs, fc, lpf, N, 0);
    
    % [핵심 수정] 현재 각도에서 7개 밴드의 시간 신호를 누적할 벡터 준비
    steered_fullband = zeros(NN_sample_len, 1);
    
    for band_idx = 1:num_bands
        % 네트워크를 통한 각도별 신호 추정 및 스케일 복원 [N x 1]
        steered_subband = predict(networks{band_idx}, dna_scan{1,band_idx}/global_max{band_idx,1});
        steered_subband = global_max{band_idx,1} * steered_subband;
        
        % 시간 영역에서 전체 대역 신호로 누적 (위에서 소리 들을 때 sum한 것과 동일)
        steered_fullband = steered_fullband + steered_subband;
        
        % --- 밴드별 독립적인 FFT 및 게인 저장 ---
        nnsig_length = length(steered_subband);
        fft_sub = fft(steered_subband);
        P2_sub = abs(fft_sub / nnsig_length);
        P1_sub = P2_sub(1:num_freq_bins);
        P1_sub(2:end-1) = 2 * P1_sub(2:end-1);
        
        % [해결] 밴드별 전용 셀 행렬의 현재 각도(행)에 대입
        BPF_NN_Gain_Matrix{band_idx}(angle_idx, :) = P1_sub(:)';%19xN
    end
    
    % --- [해결] 7개 밴드가 모두 합쳐진 Full-band 신호에 대해 FFT 수행 ---
    fft_nnsig = fft(steered_fullband);
    P2 = abs(fft_nnsig / NN_sample_len);
    P1 = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    % 전체 대역 지향성 매트릭스에 대입
    NN_Gain_Matrix(angle_idx, :) = P1(:)'; 
    
    % 기존 RMS 에너지 계산 유지
    output_energy(angle_idx) = rms(steered_fullband);
end

% --- 루프가 끝난 후 3D 플로팅 데이터 준비 ---
f = fs * (0:(num_freq_bins-1)) / NN_sample_len; 
NN_Gain_Matrix_dB = 20 * log10(NN_Gain_Matrix / max(NN_Gain_Matrix(:)));

figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);
imagesc(angles_deg, f, NN_Gain_Matrix_dB'); 
set(gca, 'YDir', 'normal'); 
colormap parula; colorbar; caxis([-80 0]); 
xlabel('Angle (Degrees)', 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontWeight', 'bold');
title('DNNBF Spatial Frequency Beampattern', 'FontWeight', 'bold');
ylim([0 fs/2]); 

for band_idx=1:num_bands
    BPF_NN_Gain_Matrix_dB = 20 * log10(BPF_NN_Gain_Matrix{1,band_idx} / max(BPF_NN_Gain_Matrix{1,band_idx},[],"all"));
    figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);
    imagesc(angles_deg, f, BPF_NN_Gain_Matrix_dB'); 
    set(gca, 'YDir', 'normal'); 
    colormap parula; colorbar; caxis([-80 0]); 
    xlabel('Angle (Degrees)', 'FontWeight', 'bold');
    ylabel('Frequency (Hz)', 'FontWeight', 'bold');
    title('DNNBF Spatial Frequency Beampattern',num2str(band_idx), 'FontWeight', 'bold');
    ylim([0 fs/2]); 
end

fprintf("DSBF max dB: %5.4f \n", max(DS_Gain_Matrix_dB(:)));
fprintf("NNBF max dB: %5.4f \n", max(NN_Gain_Matrix_dB(:)));

%%
% [수정] SI-SDR 등 평가 지표 매칭 보정
s = cell2mat(directivity_target(centered_mic_num, 1));
L = length(s);
so=cell2mat(directivity_target(:,centered_0_ongen)');
saishuonsei=sum(so,2);
DS_s_hat = saishuonsei(1:L, :);%+shifts(centered_mic_num)

% [해결] 앞서 구한 최종 복원 오디오(clean_audio)를 정답 길이 L에 맞춰 대입
NN_s_hat = clean_audio(1:L); 

interfer = noise_rir_conv_source{centered_mic_num, noise_ongen_ichi(1,1)};
[DS_SI_SDR, DS_SI_SIR, DS_SI_SAR, DS_e_targetpow, DS_e_interfpow, DS_e_artifpow] = SI_metrics(s, DS_s_hat, interfer);
[NN_SI_SDR, NN_SI_SIR, NN_SI_SAR, NN_e_targetpow, NN_e_interfpow, NN_e_artifpow] = SI_metrics(s, NN_s_hat, interfer);fprintf('\n======================================================\n');
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




