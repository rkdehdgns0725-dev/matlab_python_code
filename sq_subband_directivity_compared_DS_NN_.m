% load('sequence_NNBF_learningdata_N1.5version_1.mat')%rt60_noise
centered_mic_num=5;
directivity_target=noise_rir_conv_source;%각도별로 재생할 음원
% directivity_target=t_sound_rir_conv_source_8_18;%각도별로 재생할 음원




% 2. 테스트용 섞인 신호 준비 [N x 8]
% test_signal = t_sound_rir_conv_source(:,1) ; 
% test_signal = cell2mat(gouseionseibako(:,1)') ; 
test_signal = gouseionseibako(:,1) ; 

bpsig_to_net=BPFilt_viaLPF(test_signal,fs,fc,lpf,N,1);

tl = length(bpsig_to_net{1,1}); % 신호 길이
subband_saishu_onsei = zeros(tl, num_bands); % 서브밴드 처리된 음성 저장용 [N x 7]

for band_idx = 1:num_bands
    % 1. 해당 밴드 데이터 가져오기 및 정규화 [N x 8]
    test_input_norm = bpsig_to_net{1,band_idx} / global_max{band_idx};
    
    % 2. 🚨 [핵심 수정] 시퀀스 모델 규격에 맞게 전치 후 '셀 배열'로 포장
    % [N x 8] -> [8 x N] -> 셀 배열 래핑 { [8 x N] }
    test_input_cell = { test_input_norm' };
    
    % 3. predict 수행 (결과물은 {[1 x N]} 형태의 셀 배열로 나옵니다)
    predicted_cell = predict(networks{band_idx}, test_input_cell);
    
    % 4. 🚨 [핵심 수정] 셀 내용물을 꺼내서 다시 세로 방향 [N x 1] 열벡터로 변환
    % predicted_cell{1} 은 [1 x N] 행벡터이므로, 뒤에 전치(')를 붙여 세로로 세웁니다.
    predicted_signal = predicted_cell{1}'; 
    
    % 5. 스케일 복원 후 서브밴드 행렬에 꽂아 넣기 [N x 1]
    subband_saishu_onsei(:, band_idx) = predicted_signal * global_max{band_idx};
end
clean_audio=sum(subband_saishu_onsei,2);


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

NN_Gain_Matrix = zeros(num_sources, num_freq_bins); 
BPF_NN_Gain_Matrix = cell(1, num_bands);
for b = 1:num_bands
    BPF_NN_Gain_Matrix{b} = zeros(num_sources, num_freq_bins); 
end

% 1번부터 19번 각도까지 루프 구동
for angle_idx = 1:num_sources
    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} = directivity_target{i,angle_idx}(:);
    end
    
    dna_scan = BPFilt_viaLPF(dna_scan', fs, fc, lpf, N, 0);
    
    % 현재 각도에서 7개 밴드의 시간 신호를 누적할 벡터 준비
    steered_fullband = zeros(NN_sample_len, 1);
    
    for band_idx = 1:num_bands
        % 🚨 [차원 전처리] 정규화 및 [N x 8] -> [8 x N] 전치 후 셀 배열화
        subband_input_norm = dna_scan{1,band_idx} / global_max{band_idx,1};
        subband_input_cell = { subband_input_norm' };
        
        % 🚨 [예측 및 차원 후처리] 셀 출력 {[1 x N]} 을 다시 세로형 [N x 1] 벡터로 복원
        steered_subband_cell = predict(networks{band_idx}, subband_input_cell);
        steered_subband = steered_subband_cell{1}'; 
        
        % 스케일 원복
        steered_subband = global_max{band_idx,1} * steered_subband;
        
        % 시간 영역에서 전체 대역 신호로 누적
        steered_fullband = steered_fullband + steered_subband;
        
        % --- 밴드별 독립적인 FFT 및 게인 저장 ---
        nnsig_length = length(steered_subband);
        fft_sub = fft(steered_subband);
        P2_sub = abs(fft_sub / nnsig_length);
        P1_sub = P2_sub(1:num_freq_bins);
        P1_sub(2:end-1) = 2 * P1_sub(2:end-1);
        
        BPF_NN_Gain_Matrix{band_idx}(angle_idx, :) = P1_sub(:)'; % 19 x num_freq_bins
    end
    
    % --- 7개 밴드가 모두 합쳐진 Full-band 신호에 대해 FFT 수행 ---
    fft_nnsig = fft(steered_fullband);
    P2 = abs(fft_nnsig / NN_sample_len);
    P1 = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    NN_Gain_Matrix(angle_idx, :) = P1(:)'; 
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

% for band_idx=1:num_bands
%     BPF_NN_Gain_Matrix_dB = 20 * log10(BPF_NN_Gain_Matrix{1,band_idx} / max(BPF_NN_Gain_Matrix{1,band_idx},[],"all"));
%     figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);
%     imagesc(angles_deg, f, BPF_NN_Gain_Matrix_dB'); 
%     set(gca, 'YDir', 'normal'); 
%     colormap parula; colorbar; caxis([-80 0]); 
%     xlabel('Angle (Degrees)', 'FontWeight', 'bold');
%     ylabel('Frequency (Hz)', 'FontWeight', 'bold');
%     title('DNNBF Spatial Frequency Beampattern',num2str(band_idx), 'FontWeight', 'bold');
%     ylim([0 fs/2]); 
% end
% === 7개 밴드 통합 지향성 패턴 플롯 ===
figure('Name', 'DNNBF Subband Spatial Frequency Beampatterns', 'Position', [100, 100, 1400, 800]);

% 꼼수 핵심: 2행 12열 매트릭스를 선언하고 칸을 나눠 가집니다.
% 윗줄 4개 밴드는 각각 3칸씩 (12/4 = 3)
% 아랫줄 3개 밴드는 각각 4칸씩 (12/3 = 4)
subplot_indices = {1:3, 4:6, 7:9, 10:12, 13:15, 16:18, 19:21};
target_ticks = -90:30:90;
for band_idx = 1:num_bands
    % 1. 밴드별 데이터 정규화 및 dB 변환
    BPF_NN_Gain_Matrix_dB = 20 * log10(BPF_NN_Gain_Matrix{1,band_idx} / max(BPF_NN_Gain_Matrix{1,band_idx},[],"all"));
    
    % 2. 🚨 핵심: 2행 12열 그리드에서 지정된 인덱스 묶음을 호출하여 서브플롯 생성
    subplot(2, 12, subplot_indices{band_idx});
    
    % 3. 이미지 출력
    imagesc(angles_deg, f, BPF_NN_Gain_Matrix_dB'); 
    set(gca, 'YDir', 'normal'); 
    colormap parula; colorbar; caxis([-60 0]); 
    
set(gca, 'XTick', target_ticks);          % 눈금 위치를 -90, -60, -30, 0, 30, 60, 90으로 고정
    set(gca, 'XTickLabel', target_ticks);     % 생략 없이 해당 숫자를 그대로 텍스트로 출력
    % 4. 라벨 및 타이틀 세팅 (가독성을 위해 폰트크기 조절)
    xlabel('Angle (Deg)', 'FontSize', 9);
    ylabel('Freq (10kHz)', 'FontSize', 9);
    title(['Band ', num2str(band_idx)], 'FontWeight', 'bold', 'FontSize', 11);
    ylim([0 fs/2]); 
end
fprintf("DSBF max dB: %5.4f \n", max(DS_Gain_Matrix_dB(:)));
fprintf("NNBF max dB: %5.4f \n", max(NN_Gain_Matrix_dB(:)));

%%
% [수정] SI-SDR 등 평가 지표 매칭 보정
s = cell2mat(test_signal(centered_mic_num, 1));
L = length(s);
so=cell2mat(test_signal(:,1)');
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
fprintf([' SI-SDR |    %10' ...
    '.4f dB   |    %10.4f dB   \n'], DS_SI_SDR, NN_SI_SDR);
fprintf(' SI-SIR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SIR, NN_SI_SIR);
fprintf(' SI-SAR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SAR, NN_SI_SAR);
fprintf('------------------------------------------------------\n');
fprintf(' Target |    %10.2e      |    %10.2e      \n', DS_e_targetpow, NN_e_targetpow);
fprintf(' Interf |    %10.2e      |    %10.2e      \n', DS_e_interfpow, NN_e_interfpow);
fprintf(' Artif  |    %10.2e      |    %10.2e      \n', DS_e_artifpow, NN_e_artifpow);
fprintf('======================================================\n');

%% =========================================================================
% --- [추가] 서브밴드(대역)별 SI-SDR / SI-SIR / SI-SAR 성능 분석 ---
% =========================================================================
fprintf('\n====================================================================\n');
fprintf('        [ %s ] 각 서브밴드(대역)별 빔포밍 성능 분석 결과 \n', datestr(now, 'HH:MM:SS'));
fprintf('====================================================================\n');
fprintf(' Band  |  Metric  |   DAS (Baseline)   |   NN (Proposed)    |  改善量 (Gain)\n');
fprintf('--------------------------------------------------------------------\n');

% 주파수 대역별 루프 구동
for b = 1:num_bands
    % 1. 해당 밴드의 정답 타겟 신호 추출 (5번 마이크 기준)
    % bpsig_to_net{1, b}는 [N x 8ch]이므로 5번 마이크 성분만 추출
    s_band = bpsig_to_net{1, b}(:, centered_mic_num);
    L_band = length(s_band);
    
    % 2. 해당 밴드의 DAS(Baseline) 신호 추출
    % 입력인 test_signal(gouseionseibako) 자체를 대역 분할한 신호에서 8채널 합산
    % (만약 앞단에서 가공된 대역별 데이터가 있다면 그것을 쓰셔도 됩니다)
    so_band = bpsig_to_net{1, b}; % [N x 8ch]
    saishuonsei_band = sum(so_band, 2); % 8채널 단순 합산 (DAS)
    DS_s_hat_band = saishuonsei_band(1:L_band);
    
    % 3. 해당 밴드의 NN(Proposed) 신호 추출
    % subband_saishu_onsei에 이미 스케일 복원(*global_max)까지 완료된 밴드별 신호가 저장되어 있습니다.
    NN_s_hat_band = subband_saishu_onsei(1:L_band, b);
    
    % 4. 해당 밴드의 순수 간섭음(노이즈) 성분 추출 및 대역 분할
    % noise_rir_conv_source를 타겟 밴드와 동일하게 필터링해야 정확한 지표 계산이 가능합니다.
    % BPFilt_viaLPF 함수 특성에 맞춰 셀 구조로 래핑하여 통과시킵니다.
    interfer_raw = noise_rir_conv_source{centered_mic_num, noise_ongen_ichi(1,1)};
    interfer_cell = BPFilt_viaLPF({interfer_raw(:)}, fs, fc, lpf, N, 0); 
    interfer_band = interfer_cell{1, b}(1:L_band); % 해당 대역의 노이즈 성분 추출
    
    % 5. 밴드별 지표 계산
    [DS_SDR_b, DS_SIR_b, DS_SAR_b, ~, ~, ~] = SI_metrics(s_band, DS_s_hat_band, interfer_band);
    [NN_SDR_b, NN_SIR_b, NN_SAR_b, ~, ~, ~] = SI_metrics(s_band, NN_s_hat_band, interfer_band);
    
    % 6. 결과 출력 (각 밴드별로 3개 지표를 나란히 표시)
    fprintf('       |  SI-SDR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', ...
        DS_SDR_b, NN_SDR_b, NN_SDR_b - DS_SDR_b);
    fprintf('Band %d |  SI-SIR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', b, ...
        DS_SIR_b, NN_SIR_b, NN_SIR_b - DS_SIR_b);
    fprintf('       |  SI-SAR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', ...
        DS_SAR_b, NN_SAR_b, NN_SAR_b - DS_SAR_b);
    fprintf('--------------------------------------------------------------------\n');
end
fprintf('====================================================================\n');


