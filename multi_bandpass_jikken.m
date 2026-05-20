%% 1. 파라미터 설정
 % 샘플링 주파수 (예시)
N = 50;    % 필터 차수 (짝수 권장, 높을수록 날카로움)
% 나카지마 논문 기반 예시 주파수 (6개 경계 -> 7개 대역)
fc = [300, 500, 1000, 1500, 2000, 3000]; 
inputSignal=cell2mat(t_sound_rir_conv_source');


%% 2. LPF 설계 (fir1 사용)
% 모든 필터는 동일한 차수 N을 가져야 위상 지연이 일치하여 PR이 가능함
lpf = cell(1, length(fc));
for i = 1:length(fc)
    lpf{i} = fir1(N, fc(i)/(fs/2));
end

%% 3. 차분을 통한 서브밴드 필터(BPF) 생성
filters = cell(1, length(fc)+1);
for i=1:length(fc)+1
    if i==1
        filters{i}=lpf{i};
    elseif i<(length(fc))+1
        filters{i}=lpf{i}-lpf{i-1};
    else
        unit_impulse = zeros(1, N+1); unit_impulse(N/2 + 1) = 1; 
        filters{i} = unit_impulse - lpf{i-1};   
    end

end
% filters{1} = lpf{1};                             % 0 ~ 300Hz
% filters{2} = lpf{2} - lpf{1};                    % 300 ~ 500Hz
% filters{3} = lpf{3} - lpf{2};                    % 500 ~ 1000Hz
% filters{4} = lpf{4} - lpf{3};                    % 1000 ~ 1500Hz
% filters{5} = lpf{5} - lpf{4};                    % 1500 ~ 2000Hz
% filters{6} = lpf{6} - lpf{5};                    % 2000 ~ 3000Hz
% % 마지막 필터는 전체(델타 함수)에서 이전 LPF를 뺌
% unit_impulse = zeros(1, N+1); unit_impulse(N/2 + 1) = 1; 
% filters{7} = unit_impulse - lpf{6};              % 3000Hz ~ Nyquist

%% 4. 신호 적용 및 완전 재구성 확인
% inputSignal: [Length x 8ch] 마이크 신호라 가정
band_signals = cell(1, 7);
for b = 1:7
    band_signals{b} = filter(filters{b},1, inputSignal);%(분자, 분모(FIR)계수, 필터링할 신호)
end

% 모든 밴드 합산 (Reconstruction)
reconstructed = zeros(size(inputSignal));
for b = 1:7
    reconstructed = reconstructed + band_signals{b};
end

% 5. 오차 검증 (PR 확인)
% 필터 지연(N/2)을 고려하여 원본과 비교
delay = N/2;
error = inputSignal(1:end-delay, :) - reconstructed(delay+1:end, :);
fprintf('최대 재구성 오차: %e\n', max(abs(error(:))));

s1=inputSignal(:,1);
s2= reconstructed(:,1);
plot(s1);
hold on;
plot(s2);




% 1. 합쳐진 필터의 주파수 응답 확인
combined_filter = sum(cell2mat(filters'), 1); % 설계한 필터들 합산
[h, f] = freqz(combined_filter, 1, 1024, fs);



figure;
subplot(2,1,1);
plot(f, 20*log10(abs(h)));
title('Magnitude Response (Flatness Check)');
ylabel('Magnitude (dB)'); grid on;
ylim([-0.1 0.1]); % 아주 미세한 오차까지 확인 위해 범위 축소

subplot(2,1,2);
[gd, f_gd] = grpdelay(combined_filter, 1, 1024, fs);
plot(f_gd, gd);
title('Group Delay (Phase Linearity Check)');
ylabel('Samples'); grid on;

% 1. 시간 축을 맞춘 데이터 생성
s1_aligned = inputSignal(1:end-delay, 1);       % 원본의 뒷부분을 자름
s2_aligned = reconstructed(delay+1:end, 1);    % 재구성의 앞부분(지연)을 자름

% 2. 코히어런스 재계산
figure;
mscohere(s1_aligned, s2_aligned, hanning(512), 256, 512, fs);
title('Spectral Coherence (Aligned)');
grid on;