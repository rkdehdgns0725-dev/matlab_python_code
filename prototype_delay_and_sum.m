%%
% delay and sum--- 
% 1. 신호 합성 (오류 수정 및 최적화) ---
% mic_ch 변수 혼용 방지를 위해 num_mics로 통일

for i = 1:num_sources
    % i번째 음원 위치 (3 x 1)
    current_source = source_pos_t(:, i);

    % 중심점에서 음원을 향하는 단위 벡터 (3 x 1)
    direction_vec = current_source - co_micarray;
    co_micarray_uv = direction_vec / norm(direction_vec);

    for j = 1:num_mics
        micon_vec = current_source - mic_pos(:, j);%마이크-음원방향벡터
        delay_time = dot(micon_vec, co_micarray_uv) / c;%마이크간 시간지연
        mic_delay(j, i) = delay_time;
    end
end

delay_sample=round(fs*mic_delay);

for i = 1:num_mics
    % 혹시 모를 가로/세로 벡터 오류를 막기 위해 (:) 사용 (무조건 열벡터로)
    sig_n = noise_rir_conv_source{i, 1}(:);
    sig_t = t_sound_rir_conv_source{i, 1}(:);
    
    minimi = min(length(sig_n), length(sig_t));
    gouseionseibako{i, 1} = sig_n(1:minimi) + sig_t(1:minimi);
end

% 딜레이 보상 전 단순 합 (비교용)
hikakuyou = mean(cell2mat(gouseionseibako'), 2);


% --- 2. 올바른 Delay-and-Sum (DAS) 구현 ---
target_angle_idx = 10; 
target_delays = delay_sample(:, target_angle_idx);

% 1. 각 마이크가 '가장 늦은 마이크'에 맞추기 위해 뒤로 밀려야 하는 양 (항상 0 이상)
shifts = max(target_delays) - target_delays; 

% 2. 모든 채널의 길이를 똑같이 맞추기 위해 필요한 전체 패딩 개수
total_pad_needed = max(shifts);

dna_onsei = cell(num_mics, 1);
for i = 1:num_mics
    % 앞부분 0 패딩 (지연 보상)
    pad_front = shifts(i);
    
    % 뒷부분 0 패딩 (길이 통일용: 남은 개수만큼 채움)
    pad_back = total_pad_needed - pad_front;
    
    original_sig = gouseionseibako{i, 1};
    dna_onsei{i, 1} = [zeros(pad_front, 1); original_sig; zeros(pad_back, 1)];
end

saishuonsei = mean(cell2mat(dna_onsei'), 2);

fprintf('단순 합산 최대 피크: %f\n', max(abs(hikakuyou)));
fprintf('DAS 빔포밍 최대 피크: %f\n', max(abs(saishuonsei)));








% --- 3. 지향성 (Directivity / Spatial Spectrum) 분석 ---
disp('지향성 패턴을 계산 중입니다...');

% 19개 각도 정의 (-90도 ~ 90도, 등간격 가정)
angles_deg = linspace(-90, 90, num_sources); 
output_energy = zeros(1, 19);

% 1번부터 19번 각도까지 조향(Steering)을 돌려가며 출력 에너지 측정
for angle_idx = 1:num_sources
    % 현재 스캔 중인 각도의 딜레이 값
    scan_delays = delay_sample(:, angle_idx);
    
    % 지연 보상량 계산
    shifts = max(scan_delays) - scan_delays; 
    total_pad_needed = max(shifts);
    
    dna_scan = cell(num_mics, 1);
    for i = 1:num_mics
        pad_front = shifts(i);
        pad_back = total_pad_needed - pad_front;
        % original_sig = gouseionseibako{i, 1};%분석할 음원
        original_sig = t_sound_rir_conv_source{i, 1};%분석할 음원

        dna_scan{i, 1} = [zeros(pad_front, 1); original_sig; zeros(pad_back, 1)];
    end
    
    % 스캔된 방향으로의 출력 신호
    steered_signal = mean(cell2mat(dna_scan'), 2);
    
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
figure('Name', 'DAS Beamformer Directivity', 'Position', [100, 100, 1000, 400]);

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