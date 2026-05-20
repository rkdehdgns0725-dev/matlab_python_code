%%

% 19개 각도 정의
angles_deg = linspace(-90, 90, num_sources); 
output_energy = zeros(1, num_sources);

% [중요] 루프 진입 전 주파수 빈(Bin) 개수 사전 확정
% 임의로 1번째 채널 신호 길이를 파악하여 크기를 지정합니다.
sample_len = length(noise_rir_conv_source{1, 1}); 
num_freq_bins = floor(sample_len/2) + 1;

% 1번부터 19번 각도까지 루프 구동
for angle_idx = 1:num_sources
    dna_scan = cell(1, num_mics);
    for i = 1:num_mics
        dna_scan{1,i} = noise_rir_conv_source{i,angle_idx}(:);
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
    Gain_Matrix(angle_idx, :) = P1(:)'; 
    
    % 기존 RMS 에너지 계산 유지
    output_energy(angle_idx) = rms(steered_signal);
end

% --- 루프가 끝난 후 3D 플로팅 데이터 준비 ---
f = fs * (0:(num_freq_bins-1)) / sample_len; % 주파수 축 배열 (1 x num_freq_bins)

% [논문 표준] 주파수 게인을 dB 스케일로 변환 및 최고점을 0dB로 정규화
% 리니어 스케일로 그리면 감쇄 대역(Null)이 평평하게 묻혀서 안 보입니다.
Gain_Matrix_dB = 20 * log10(Gain_Matrix / max(Gain_Matrix(:)));

% 각도(angles_deg)와 주파수(f) 1차원 배열을 2차원 그리드 평면 좌표로 변환
[Freq_Grid, Angle_Grid] = meshgrid(f, angles_deg);

% --- 3D Surface 그래프 그리기 ---
figure('Name', '3D Spatial Frequency Response', 'Position', [150, 150, 900, 600]);
mesh(Angle_Grid, Freq_Grid, Gain_Matrix_dB);

% 그래프 가독성을 높이는 고급 시각화 세팅
shading interp;      % 그리드 라인을 지우고 색상을 부드럽게 보간 (논문 필수 수식)
colormap jet;        % 감쇄는 파란색, 피크는 빨간색으로 표현하는 칼라맵
colorbar;            % 우측에 dB 색상 바 표시
caxis([-30 0]);      % 색상 바의 범위를 -30dB ~ 0dB로 고정

% 축 레이블 및 타이틀 세팅
xlabel('Angle (Degrees)', 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontWeight', 'bold');
zlabel('Normalized Gain (dB)', 'FontWeight', 'bold');
title('3D Spatial Frequency Beampattern (NN Proposed)', 'FontSize', 12, 'FontWeight', 'bold');

% 관심 영역(300Hz ~ 3000Hz)으로 그래프 축 제한 시키기
xlim([-90 90]);
ylim([0 fs/2]);    % 사용자님의 목적 주파수 대역인 300~3000Hz만 타겟팅
zlim([-30 2]);       % 아래 노이즈 바닥을 잘라내고 피크 강조





%%
% 3D 뷰 카메라 각도 최적화 (시점 조절: Azimuth, Elevation)
% view(-35, 40); 
grid on;

figure('Name', '2D Spatial Frequency Response', 'Position', [150, 150, 600, 500]);

% 3D가 아닌 2D 이미지 형태로 출력 (위에서 내려다봄)
imagesc(angles_deg, f, Gain_Matrix_dB'); 

% y축(주파수)이 아래에서 위로 올라가도록 방향 뒤집기 (imagesc의 기본 특성 보정)
set(gca, 'YDir', 'normal'); 

% 색상은 기본 parula(최신 매트랩 논문 표준) 또는 gray 사용
colormap parula; % 또는 colormap gray;
colorbar;
caxis([-50 0]); % -30dB 이하는 뭉뚱그림

xlabel('Angle (Degrees)', 'FontWeight', 'bold');
ylabel('Frequency (Hz)', 'FontWeight', 'bold');
title('Spatial Frequency Beampattern', 'FontWeight', 'bold');
ylim([0 fs/2]); % 타겟 주파수 대역
