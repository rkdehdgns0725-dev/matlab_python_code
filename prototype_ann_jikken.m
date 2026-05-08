
load('NNBF_learningdata.mat')
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

soundsc(clean_audio, fs);
% soundsc(saishuonsei,fs);



%%
% 화이트노이즈에 대한 지향성 플롯

% --- 3. 지향성 (Directivity / Spatial Spectrum) 분석 ---
disp('지향성 패턴을 계산 중입니다...');

% 19개 각도 정의 (-90도 ~ 90도, 등간격 가정)
angles_deg = linspace(-90, 90, num_sources); 
output_energy = zeros(1, num_sources);

% 1번부터 19번 각도까지 조향(Steering)을 돌려가며 출력 에너지 측정
for angle_idx = 1:num_sources

    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} =t_sound_rir_conv_source_8_18{i,angle_idx}(:)/global_max;%target sound
    end
    
    % 스캔된 방향으로의 출력 신호
    steered_signal = predict(net,cell2mat(dna_scan));
    steered_signal=global_max*steered_signal;
    % % 추가: 2000Hz 이상 고주파수 대역만 필터링해서 확인해보기
    % steered_signal_high = highpass(steered_signal, 2000, fs);
    % 
    % % 해당 방향에서 픽업된 신호의 에너지(RMS) 저장 (필터링된 신호 사용)
    % output_energy(angle_idx) = rms(steered_signal_high);

    % % 해당 방향에서 픽업된 신호의 에너지(RMS) 저장
    output_energy(angle_idx) = rms(steered_signal);
end

% 에너지를 dB 스케일로 변환 및 정규화 (가장 큰 피크를 0dB로)
output_energy_dB = 20 * log10(output_energy / max(output_energy));

% --- 그래프 출력 ---
figure('Name', 'DNN Beamformer Directivity', 'Position', [100, 100, 1000, 400]);

% 1. 직교 좌표계 (Cartesian Plot)
subplot(1, 2, 1);
plot(angles_deg, output_energy_dB, '-o', 'LineWidth', 2, 'Color', '#0072BD');
grid on;
xlabel('Steering Angle (Degrees)');
ylabel('Normalized Output Power (dB)');
title('Spatial Spectrum (Cartesian)');
ylim([-30 2]); % -30dB까지만 표시 (필요시 조절)
xticks(-90:30:90);

% 2. 극 좌표계 (Polar Plot) - 음향/안테나 논문 표준 포맷
subplot(1, 2, 2);
angles_rad = deg2rad(angles_deg);
polarplot(angles_rad, output_energy_dB, '-o', 'LineWidth', 2, 'Color', '#D95319');
ax = gca;
ax.ThetaZeroLocation = 'top'; % 정면(0도)을 12시 방향으로
ax.ThetaDir = 'clockwise';    % 각도를 시계방향으로 전개
rlim([-30 0]);                % dB의 범위를 -30 ~ 0으로 설정
title('Beampattern (Polar)');