Fs = 96000;
s = 0.1;
filename=['C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\matlab_code\ongen\white_gaussian_noise.wav']
noisysound=audioread(filename);
L=length(noisysound);

% FFT 계산
Y = fft(noisysound);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

plot(f, P1);
grid on;
title(['Channel']);
xlabel('주파수 (Hz)');
ylabel('Magnitude');
xlim([0 Fs/2]); % 0~100Hz 범위 표시

% 전체 그래프 제목 설정
sgtitle(sprintf('FFT Spectrum for All "%s" Dataset', 'a'), 'FontSize', 14, 'FontWeight', 'bold');
