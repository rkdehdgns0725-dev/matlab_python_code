function [band_signals]=BPFilt_viaLPF(t_sound_rir_conv_source,fs,fc,lpf,N,plot_on)
    % inputSignal: [Length x 8ch] 행렬 변환
    inputSignal = cell2mat(t_sound_rir_conv_source');
    
    num_bands = length(fc) + 1;
    band_signals = cell(1, num_bands);
    
    % --- 핵심: 신호 영역에서 차분 수행 (filtfilt 전용 PR 기법) ---
    
    % 1. 첫 번째 저역 대역 (LPF 1단계 통과)
    band_signals{1} = filtfilt(lpf{1}, 1, inputSignal);
    
    % 2. 중간 밴드패스 대역 (위 단계 LPF 신호 - 아래 단계 LPF 신호)
    for b = 2:num_bands-1
        % 각각 순수 LPF로 filtfilt를 한 후 신호를 빼줍니다.
        low_high = filtfilt(lpf{b}, 1, inputSignal);
        low_low  = filtfilt(lpf{b-1}, 1, inputSignal);
        band_signals{b} = low_high - low_low; 
    end
    
    % 3. 마지막 고역 대역 (원본 신호 - 가장 높은 LPF 신호)
    % unit_impulse를 설계할 필요 없이 원본에서 빼면 정확히 HPF 효과가 납니다.
    band_signals{num_bands} = inputSignal - filtfilt(lpf{end}, 1, inputSignal);
    
    %% [검증] 이렇게 하면 다 더했을 때 원본과 100% 일치하는지 확인 가능
    % reconstructed = zeros(size(inputSignal));
    % for b = 1:num_bands, reconstructed = reconstructed + band_signals{b}; end
    % max(abs(inputSignal - reconstructed))% -> 0에 수렴해야 정상!
    %% 4. 신호 적용 및 완전 재구성 확인
    
    if plot_on==true
        figure;
        for i=1:length(band_signals)
            subplot(length(band_signals),1,i)
        
            bpsig=band_signals{1,i};
            L=length(bpsig);
        
            % FFT 계산
            Y = fft(bpsig);
            P2 = abs(Y/L);
            P1 = P2(1:floor(L/2)+1);
            P1(2:end-1) = 2*P1(2:end-1);
            f = fs*(0:(L/2))/L;
            
            plot(f, P1);
            grid on;
            title(['bandpassed',num2str(i),'band- signal']);
            xlabel('frequency (10kHz)');
            ylabel('Magnitude');
            xlim([0 fs/2]); % 0~100Hz 범위 표시
        end
    end
end