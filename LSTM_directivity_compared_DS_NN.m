% load('sequence_NNBF_learningdata_N1.5version_1.mat') % 학습된 모델 로드
centered_mic_num = 5;
directivity_target = noise_rir_conv_source; 

test_signal = gouseionseibako(:,1); 
bpsig_to_net = BPFilt_viaLPF(test_signal, fs, fc, lpf, N, 1);
tl = length(bpsig_to_net{1,1}); 
subband_saishu_onsei = zeros(tl, num_bands); 
test_chunkSize = 128; 

% 1. 전체 시간 축에 대해 슬라이딩 윈도우 행렬을 미리 통째로 만듭니다. (루프 최소화)
num_samples_to_predict = tl - test_chunkSize + 1;

fprintf('데이터 블록을 고속 생성 중입니다...\n');
% 모든 시점의 [8 x 128] 조각들을 담을 거대한 셀 배열 사전 할당
% 밴드가 바뀌어도 입력 데이터의 인덱스 구조는 같으므로 루프 밖에서 1회만 수행
X_test_cell = cell(num_samples_to_predict, 1);

% 이 루프는 단순 인덱싱이므로 CPU에서 몇 초 만에 끝납니다.
for idx = test_chunkSize:tl
    c_idx = idx - test_chunkSize + 1;
    % 해당 시점의 [128 x 8] 추출 -> [8 x 128]로 전치하여 셀에 격납
    % (참고: 정규화는 밴드별로 다르므로 여기선 날것의 bpsig_to_net을 기준으로 인덱싱 틀만 잡음)
end

fprintf('7개 밴드 고속 병렬 predict 시작 (약 수 초 소요)...\n');
for band_idx = 1:num_bands
    % 밴드별 정규화
    test_input_norm = bpsig_to_net{1,band_idx} / global_max{band_idx};
    
    % 거대한 셀 배열에 정규화된 데이터 채워 넣기
    X_band_cell = cell(num_samples_to_predict, 1);
    for idx = test_chunkSize:tl
        c_idx = idx - test_chunkSize + 1;
        X_band_cell{c_idx, 1} = test_input_norm(idx - test_chunkSize + 1 : idx, :)';
    end
    
    % 🚨 [마법의 구간] 만 단위의 셀 배열을 predict에 통째로 던집니다.
    % 매트랩 내부 C++ 엔진이 GPU 배치를 알아서 쪼개어 초고속으로 병렬 처리합니다.
    % 결과물은 [num_samples_to_predict x 1] 크기의 수치 행렬로 한 번에 나옵니다.
    pred_vals = predict(networks{band_idx}, X_band_cell);
    
    % 결과 대입 및 정규화 복원
    predicted_band_signal = zeros(tl, 1);
    % 처음 부족한 구간 채우기
    predicted_band_signal(1:test_chunkSize-1) = test_input_norm(1:test_chunkSize-1, center_mic);
    % predict된 알맹이 통째로 꽂아 넣기
    predicted_band_signal(test_chunkSize:end) = pred_vals;
    
    subband_saishu_onsei(:, band_idx) = predicted_band_signal * global_max{band_idx};
    fprintf('%d번 밴드 초고속 복원 완료!\n', band_idx);
end

clean_audio = sum(subband_saishu_onsei, 2);
% 6. 오디오 재생 확인
soundsc(clean_audio, fs);

%% =========================================================================
% --- [공정 평가] 대역별 및 전체 SI-SDR 계산단 ---
% =========================================================================
s = cell2mat(test_signal(centered_mic_num, 1));
L = length(s);

so = cell2mat(test_signal(:,1)');
saishuonsei = sum(so, 2);
DS_s_hat = saishuonsei(1:L, :); % DAS 통짜 신호
NN_s_hat = clean_audio(1:L);    % NN 복원 통짜 신호

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