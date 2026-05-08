% === 3000~5000Hz 대역 제한 랜덤 노이즈 생성 코드 === %
clear; clc;

% 1. 기본 설정 (원본 코드 환경과 동일하게 맞춤)
Fs = 96000;           % 샘플링 주파수 [Hz]
duration = 5;         % 생성할 오디오 길이 (5초면 시뮬레이션에 충분함)
t = 0:1/Fs:duration-1/Fs;

% 2. 화이트 노이즈(모든 주파수 대역이 섞인 소리) 생성
white_noise = randn(length(t), 1);

% 3. 대역통과 필터(Bandpass Filter) 적용: 3000 ~ 5000 Hz만 통과
% matlab의 내장 함수 bandpass를 사용합니다.
band_limited_noise = bandpass(white_noise, [3000 5000], Fs);

% 4. 클리핑 방지를 위한 정규화 (진폭을 -0.9 ~ 0.9 사이로 맞춤)
max_val = max(abs(band_limited_noise));
band_limited_noise = (band_limited_noise / max_val) * 0.9;

% 5. 오디오 파일로 저장
% 원본 코드가 찾는 파일명 그대로 저장합니다.
filename = 'ランダムノイズ_帯域3000~5000Hz_フラット.wav';
audiowrite(filename, band_limited_noise, Fs);

disp('노이즈 파일이 성공적으로 생성되었습니다!');
disp(['저장된 파일명: ', filename]);