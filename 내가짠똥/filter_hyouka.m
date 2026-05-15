close all;

crossFilter=crossoverFilter(4,[400,800,1600,3200]);
window_len=2048;%coherence 
window_len_2=2*fs;%impulse
ov_p=0.5;%오버랩율


impulse = [1; zeros(window_len_2-1, 1)]; 

% 2. フィルタを通過させて各帯域の出力を得る
% ※ crossoverFilterは出力数が帯域数と同じである必要があります
[b1, b2, b3, b4, b5] = crossFilter(impulse);%각 필터에 해당하는 임펄스응답
    
% 3. 全帯域の出力を加算（これが完全再構成後のインパルス応答になります）
summed_impulse = b1 + b2 + b3 + b4 + b5;%필터의 임펄스 응답



% Apply the crossover filter to a signal
[band1, band2, band3, band4, band5]=crossFilter(t_sound);
reconst=zeros(size(band1));%Crossfilter가 적용된걸 재구성-음원
for i=[band1, band2, band3, band4, band5]
    rms(i)
    reconst=reconst+i;
end

[band1, band2, band3, band4, band5]=crossFilter(noise_mono);
reconst_WGN=zeros(size(band1));%Crossfilter가 적용된걸 재구성-노이즈
for i=[band1, band2, band3, band4, band5]
    rms(i)
    reconst_WGN=reconst_WGN+i;
end
rms(reconst)
rms(t_sound)
% [h_cross, f_cross] = freqz(crossFilter(t_sound), 1, 1024, fs);


delay_crossover = finddelay(t_sound, reconst)
d=0;

% soundsc(reconst,fs)

%%
%원본 코드에서 했던 방식
bandpass_fc= [300, 500, 1000, 1500, 2000, 3000]; %밴드패스에 사용되는 주파수
fc=bandpass_fc;
bf_signal = zeros(length(t_sound), 7);
for i = 1:5
     bf_signal(:, i) = bandpass(t_sound, [fc(i), fc(i+1)],fs);
end
bf_signal_300_3000=sum(bf_signal,2);%각각의 밴드패스 된 신호의 합:300~3000Hz
bf_signal_tsound=bandpass(t_sound,[300, 3000],fs);%한개의 밴드패스로 필터링된 신호:300~3000Hz
error_bf=max(bf_signal_300_3000-bf_signal_tsound,[],'all')%최대오차

plot(bf_signal_300_3000,"DisplayName",'subbanded and reconstructed 300-3000Hz');
hold on;
plot(bf_signal_tsound,'DisplayName','300-3000Hz BPF');
legend;

figure;
[CBF,FBF]=mscohere(bf_signal_300_3000,bf_signal_tsound, hanning(window_len), ov_p*window_len, window_len, fs);
plot(FBF,10*log10(CBF),'DisplayName','reconstructed after subbanding','LineWidth',1.5);
xlim([290,3100]);
ylim([-0.5,0.001])
hold on; % 기존 그래프 위에 겹쳐 그려야 하므로 필수
legend;

for i = 1:length(bandpass_fc)
    % xline(위치, '선모양색상', '설명')
    xline(bandpass_fc(i), 'r--', 'LineWidth', 1.5,'HandleVisibility', 'off'); 
end
ylabel('Magnitude (dB)'); grid on;
title('Spectral Coherence Comparison');
hold off;



% 1. 두 신호의 오차 계산 (절대값 사용)
raw_diff = bf_signal_300_3000 - bf_signal_tsound;

% 2. 시간 어긋남(Delay) 확인
d = finddelay(bf_signal_tsound, bf_signal_300_3000);
fprintf('두 신호의 시간 차이: %d 샘플\n', d);

% 3. 지연을 보정한 진짜 오차 계산
if d > 0
    aligned_diff = bf_signal_300_3000(d+1:end) - bf_signal_tsound(1:end-d);
elseif d < 0
    d = abs(d);
    aligned_diff = bf_signal_300_3000(1:end-d) - bf_signal_tsound(d+1:end);
else
    aligned_diff = raw_diff;
end

error_bf = max(abs(aligned_diff), [], 'all');
fprintf('최종 보정된 오차: %e\n', error_bf);


%% 1. 파라미터 설정
 % 샘플링 주파수 (예시)
N = 50;    % 필터 차수 (짝수 권장, 높을수록 날카로움)
% 나카지마 논문 기반 예시 주파수 (6개 경계 -> 7개 대역)
fc = [300, 500, 1000, 1500, 2000, 3000]; 
inputSignal=t_sound.*ones(1,mic_ch);
% inputSignal=cell2mat(t_sound_rir_conv_source');


%% 2. LPF 설계 (fir1 사용)
% 모든 필터는 동일한 차수 N을 가져야 위상 지연이 일치하여 PR이 가능함
lpf = cell(1, length(fc));
for i = 1:length(fc)
    lpf{i} = fir1(N, fc(i)/(fs/2));
end

%% 差分フィルタ
filters = cell(1, 7);
filters{1} = lpf{1};                             % 0 ~ 300Hz
filters{2} = lpf{2} - lpf{1};                    % 300 ~ 500Hz
filters{3} = lpf{3} - lpf{2};                    % 500 ~ 1000Hz
filters{4} = lpf{4} - lpf{3};                    % 1000 ~ 1500Hz
filters{5} = lpf{5} - lpf{4};                    % 1500 ~ 2000Hz
filters{6} = lpf{6} - lpf{5};                    % 2000 ~ 3000Hz

unit_impulse = zeros(1, N+1); unit_impulse(N/2 + 1) = 1; 
filters{7} = unit_impulse - lpf{6};              % 3000Hz ~ Nyquist
% 마지막 필터는 전체(델타 함수)에서 이전 LPF를 뺌

%% 4. 신호 적용 및 완전 재구성 확인
% inputSignal: [Length x 8ch] 마이크 신호라 가정
band_signals = cell(1, 7);
for b = 1:7
    band_signals{b} = filter(filters{b}, 1, inputSignal);
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












%%
% plot
s1= inputSignal(1:end-delay,1);
s2= reconstructed(1+delay:end,1);%lpf_7
s3= reconst;%CrossFilter
figure;
subplot(2,1,1);
plot(s1,"DisplayName",'Input_signal');
hold on;
plot(s2,"DisplayName",'reconstructed via LPF_7');
xlim([fs*0.3,fs*0.4]);
legend;

subplot(2,1,2);
plot(s1,"DisplayName",'Input_signal');
hold on;
plot(s3,"DisplayName",'reconstructed via CrossOver Filter order');
xlim([fs*0.3,fs*0.4]);
legend;





% 1. 합쳐진 필터의 주파수 응답 확인
combined_filter = sum(cell2mat(filters'), 1); % 설계한 필터들 합산
[h, f] = freqz(combined_filter, 1, 1024, fs);
[h_cross, f_cross] = freqz(summed_impulse, 1, 1024, fs);


figure;
subplot(2,1,1);
plot(f, 20*log10(abs(h)),'DisplayName','Lowpass_7');
hold on;
plot(f_cross, 20*log10(abs(h_cross)),'DisplayName','CrossOver Filter');
legend;

title('Magnitude Response (Flatness Check)');
ylabel('Magnitude (dB)'); grid on;
ylim([-0.1 0.1]); % 아주 미세한 오차까지 확인 위해 범위 축소

subplot(2,1,2);
[gd, f_gd] = grpdelay(combined_filter, 1, 1024, fs);
[gd_cross, f_gd_cross] = grpdelay(summed_impulse, 1, 1024, fs);

plot(f_gd, gd,'DisplayName','Lowpass_7');
hold on;
plot(f_gd_cross,gd_cross,'DisplayName','CrossOver Filter')
legend;
title('Group Delay (Phase Linearity Check)');
ylabel('Samples'); grid on;

% 1. 시간 축을 맞춘 데이터 생성
s1_aligned = inputSignal(1:end-delay, 1);       % 원본의 뒷부분을 자름
s2_aligned = reconstructed(delay+1:end, 1);    % 재구성의 앞부분(지연)을 자름

% 2. 코히어런스 재계산
frequency_resolution=fs/window_len

% 1. 데이터 계산 (출력 인자를 지정해서 데이터를 변수에 저장합니다)
[C1, f1] = mscohere(s1_aligned, s2_aligned, hanning(window_len), ov_p*window_len, window_len, fs);
[C2, f2] = mscohere(t_sound(1:end,1), reconst(1:end,1), hanning(window_len), ov_p*window_len, window_len, fs);
[C3, f3] = mscohere(noise_mono(1:end,1), reconst_WGN, hanning(window_len), ov_p*window_len, window_len, fs);

% 2. 직접 Plot 그리기
figure;
plot(f1, C1, 'b', 'LineWidth', 1.5, 'DisplayName', 'Proposed FIR (Speech)'); 
hold on;
plot(f2, C2, 'r', 'LineWidth', 1.2, 'DisplayName', 'Crossover IIR (Speech)');
plot(f3, C3, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Crossover IIR (WGN)'); % WGN은 검정 점선으로 구분

% 3. 그래프 꾸미기
grid on;
xlim([0,fs/2]);
ylim([0.94 1.02]); % 코히어런스는 0~1 사이이므로 범위를 고정
xlabel('Frequency (Hz)');
ylabel('Coherence');
title('Spectral Coherence Comparison (Aligned)');

% 4. 범례 표시 (DisplayName을 자동으로 불러옵니다)
legend('Location', 'southwest');
