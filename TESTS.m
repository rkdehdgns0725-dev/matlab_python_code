% load('sequence_NNBF_learningdata_N1.5version_1.mat') % 모델 로드 상태 점검
centered_mic_num = 5;
directivity_target = noise_rir_conv_source; % 각도별 재생할 음원

test_signal = gouseionseibako(:,1); 
bpsig_to_net = BPFilt_viaLPF(test_signal, fs, fc, lpf, N, 1);
tl = length(bpsig_to_net{1,1}); 
subband_saishu_onsei = zeros(tl, num_bands); 
test_chunkSize = 128; 

% 🚨 [핵심 고속화] 1샘플씩 전진하는 모든 슬라이딩 윈도우의 개수 계산
num_total_slides = tl - test_chunkSize + 1;

fprintf('고속 슬라이딩 윈도우 데이터 블록을 생성 중입니다...\n');
% 모든 시점의 [8 x 128] 조각들을 한 번에 담을 거대한 셀 배열 사전 할당
X_test_cell_template = cell(num_total_slides, 1);

fprintf('7개 밴드 빈틈없는 초고속 복원 시작...\n');
for band_idx = 1:num_bands
    test_input_norm = bpsig_to_net{1,band_idx} / global_max{band_idx};
    
    % 밴드별 정규화 데이터를 셀 배열에 빈틈없이 채우기
    X_band_cell = X_test_cell_template;
    for idx = test_chunkSize:tl
        c_idx = idx - test_chunkSize + 1;
        % 1샘플씩 밀어가며 모든 과거 128샘플 조각들을 담음
        X_band_cell{c_idx, 1} = test_input_norm(idx - test_chunkSize + 1 : idx, :)';
    end
    
    % 🚨 [마법의 구간] 1샘플씩 슬라이딩된 수만 개의 데이터를 통째로 predict 호출
    % 결과물은 빈 공간 없이 연속된 [num_total_slides x 1] 크기의 완벽한 오디오 신호입니다.
    pred_vals = predict(networks{band_idx}, X_band_cell);
    
    % 시간 축 신호에 빈틈없이 대입
    predicted_band_signal = zeros(tl, 1);
    % 앞부분 부족한 구간은 센터 마이크 신호로 임시 보간
    predicted_band_signal(1:test_chunkSize-1) = test_input_norm(1:test_chunkSize-1, center_mic);
    % 연속된 예측 데이터 통째로 꽂아 넣기 (찌지직 소리 원천 차단)
    predicted_band_signal(test_chunkSize:end) = pred_vals;
    
    subband_saishu_onsei(:, band_idx) = predicted_band_signal * global_max{band_idx};
    fprintf('  -> Band %d 완벽 복원 완료\n', band_idx);
end

clean_audio = sum(subband_saishu_onsei, 2);
% 오디오 출력 확인
soundsc(clean_audio, fs);
% =========================================================================
% --- 3. [최종 고속 결정판] 1샘플 슬라이딩 배치 기반 빔패턴 연산 ---
% =========================================================================
fprintf('\n주파수 왜곡을 제거한 1샘플 슬라이딩 배치 빔패턴 연산을 시작합니다...\n');
angles_deg = linspace(-90, 90, num_sources); 
output_energy = zeros(1, num_sources);

NN_sample_len = length(directivity_target{1, 1}); 
num_freq_bins = floor(NN_sample_len/2) + 1;

% 🚨 1샘플 슬라이딩 시 생성되는 총 청크 개수 계산
test_chunkSize = 128;
num_spatial_slides = NN_sample_len - test_chunkSize + 1;

NN_Gain_Matrix = zeros(num_sources, num_freq_bins); 
BPF_NN_Gain_Matrix = cell(1, num_bands);
for b = 1:num_bands
    BPF_NN_Gain_Matrix{b} = zeros(num_sources, num_freq_bins); 
end

fprintf('전체 각도 소스에 대한 일괄 서브밴드 필터링 수행 중...\n');
all_angles_scan = cell(num_sources * num_mics, 1); 
for angle_idx = 1:num_sources
    for i = 1:num_mics
        target_row = (angle_idx - 1) * num_mics + i;
        all_angles_scan{target_row, 1} = directivity_target{i, angle_idx}(:);
    end
end

dna_scan_filtered = BPFilt_viaLPF(all_angles_scan, fs, fc, lpf, N, 0);

% 1번부터 19번 각도 분석 루프
for angle_idx = 1:num_sources
    steered_fullband = zeros(NN_sample_len, 1);
    ch_start_idx = (angle_idx - 1) * num_mics + 1;
    ch_end_idx = angle_idx * num_mics;
    
    for band_idx = 1:num_bands
        subband_raw = dna_scan_filtered{1, band_idx}(:, ch_start_idx:ch_end_idx); 
        subband_input_norm = subband_raw / global_max{band_idx, 1};
        
        % 🚨 [핵심 변경] 이 밴드/각도 한 조각에 대해서만 1샘플 슬라이딩 셀 배열 밀집 생성
        % 루프가 끝나면 메모리에서 즉시 소멸하므로 메모리 OOM이 절대 나지 않습니다.
        X_spatial_cell = cell(num_spatial_slides, 1);
        for c = 1:num_spatial_slides
            X_spatial_cell{c, 1} = subband_input_norm(c : c + test_chunkSize - 1, :)'; 
        end
        
        % 단 1번의 predict 호출로 연속된 모든 샘플 정답 고속 획득
        pred_spatial_vals = predict(networks{band_idx}, X_spatial_cell);
        
        % 🚨 징검다리 대입과 interp1을 과감히 버리고, 빈틈없이 100% 촘촘하게 신호 복원
        steered_subband = zeros(NN_sample_len, 1);
        steered_subband(1:test_chunkSize-1) = subband_input_norm(1:test_chunkSize-1, center_mic);
        steered_subband(test_chunkSize:end) = pred_spatial_vals;
        
        % 스케일 복원 및 누적
        steered_subband = global_max{band_idx,1} * steered_subband;
        steered_fullband = steered_fullband + steered_subband;
        
        % 밴드별 FFT 분석 데이터 적재 (이제 제 고유 주파수 층으로 올라갑니다)
        nnsig_length = length(steered_subband);
        fft_sub = fft(steered_subband);
        P2_sub = abs(fft_sub / nnsig_length);
        P1_sub = P2_sub(1:num_freq_bins);
        P1_sub(2:end-1) = 2 * P1_sub(2:end-1);
        
        BPF_NN_Gain_Matrix{band_idx}(angle_idx, :) = P1_sub(:)';
    end
    
    % Full-band 종합 신호 FFT
    fft_nnsig = fft(steered_fullband);
    P2 = abs(fft_nnsig / NN_sample_len);
    P1 = P2(1:num_freq_bins);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    NN_Gain_Matrix(angle_idx, :) = P1(:)'; 
    output_energy(angle_idx) = rms(steered_fullband);
end
fprintf('✅ 고해상도 빔패턴 공간 분석 완료!\n');
%% =========================================================================
% --- 4. 지향성 패턴(Beampattern) 시각화 플롯 출력 ---
% =========================================================================
f = fs * (0:(num_freq_bins-1)) / NN_sample_len; 

% 7개 밴드 통합 지향성 패턴 멀티 플롯 서브비주얼화
figure('Name', 'DNNBF Subband Spatial Frequency Beampatterns', 'Position', [100, 100, 1400, 800]);
subplot_indices = {1:3, 4:6, 7:9, 10:12, 13:15, 16:18, 19:21};
target_ticks = -90:30:90;

for band_idx = 1:num_bands
    BPF_NN_Gain_Matrix_dB = 20 * log10(BPF_NN_Gain_Matrix{1,band_idx} / max(BPF_NN_Gain_Matrix{1,band_idx},[],"all"));
    
    subplot(2, 12, subplot_indices{band_idx});
    imagesc(angles_deg, f, BPF_NN_Gain_Matrix_dB'); 
    set(gca, 'YDir', 'normal'); 
    colormap parula; colorbar; caxis([-60 0]); 
    
    set(gca, 'XTick', target_ticks);          
    set(gca, 'XTickLabel', target_ticks);     
    xlabel('Angle (Deg)', 'FontSize', 9);
    ylabel('Freq (Hz)', 'FontSize', 9);
    title(['Band ', num2str(band_idx)], 'FontWeight', 'bold', 'FontSize', 11);
    ylim([0 fs/2]); 
end

%% =========================================================================
% --- 5. [종합 평가] 전대역 및 대역별 SI-SDR / SI-SIR / SI-SAR 분석 ---
% =========================================================================
s = cell2mat(test_signal(centered_mic_num, 1));
L = length(s);
so = cell2mat(test_signal(:,1)');
saishuonsei = sum(so, 2);
DS_s_hat = saishuonsei(1:L, :);
NN_s_hat = clean_audio(1:L); 

interfer = noise_rir_conv_source{centered_mic_num, noise_ongen_ichi(1,1)};
[DS_SI_SDR, DS_SI_SIR, DS_SI_SAR, ~, ~, ~] = SI_metrics(s, DS_s_hat, interfer);
[NN_SI_SDR, NN_SI_SIR, NN_SI_SAR, ~, ~, ~] = SI_metrics(s, NN_s_hat, interfer);

fprintf('\n======================================================\n');
fprintf('        [ %s ] Full-band 빔포밍 성능 종합 결과 \n', datestr(now, 'HH:MM:SS'));
fprintf('======================================================\n');
fprintf(' Metric |   DAS (Baseline)   |   NN (Proposed)    \n');
fprintf('------------------------------------------------------\n');
fprintf(' SI-SDR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SDR, NN_SI_SDR);
fprintf(' SI-SIR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SIR, NN_SI_SIR);
fprintf(' SI-SAR |    %10.4f dB   |    %10.4f dB   \n', DS_SI_SAR, NN_SI_SAR);
fprintf('======================================================\n');

fprintf('\n====================================================================\n');
fprintf('        各 서브밴드(대역)별 상세 성능 분석 결과 \n');
fprintf('====================================================================\n');
fprintf(' Band  |  Metric  |   DAS (Baseline)   |   NN (Proposed)    |  改善量 (Gain)\n');
fprintf('--------------------------------------------------------------------\n');

for b = 1:num_bands
    s_band = bpsig_to_net{1, b}(:, centered_mic_num);
    L_band = length(s_band);
    
    so_band = bpsig_to_net{1, b}; 
    saishuonsei_band = sum(so_band, 2); 
    DS_s_hat_band = saishuonsei_band(1:L_band);
    NN_s_hat_band = subband_saishu_onsei(1:L_band, b);
    
    interfer_cell = BPFilt_viaLPF({interfer(:)}, fs, fc, lpf, N, 0); 
    interfer_band = interfer_cell{1, b}(1:L_band); 
    
    [DS_SDR_b, DS_SIR_b, DS_SAR_b, ~, ~, ~] = SI_metrics(s_band, DS_s_hat_band, interfer_band);
    [NN_SDR_b, NN_SIR_b, NN_SAR_b, ~, ~, ~] = SI_metrics(s_band, NN_s_hat_band, interfer_band);
    
    fprintf('       |  SI-SDR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', DS_SDR_b, NN_SDR_b, NN_SDR_b - DS_SDR_b);
    fprintf('Band %d |  SI-SIR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', b, DS_SIR_b, NN_SIR_b, NN_SIR_b - DS_SIR_b);
    fprintf('       |  SI-SAR  |    %10.4f dB   |    %10.4f dB   |   %10.4f dB\n', ...
        DS_SAR_b, NN_SAR_b, NN_SAR_b - DS_SAR_b);
    fprintf('--------------------------------------------------------------------\n');
end
fprintf('====================================================================\n');