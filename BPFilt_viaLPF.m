function [band_signals]=BPFilt_viaLPF(t_sound_rir_conv_source,fs,fc,lpf,N,plot_on)
    %% 1. 파라미터 설정
     % 샘플링 주파수 (예시)
    % N = 50;    % 필터 차수 (짝수 권장, 높을수록 날카로움)
    % 나카지마 논문 기반 예시 주파수 (6개 경계 -> 7개 대역)
    % fc = [500, 1000, 2000, 3000, 6000, 12000]; 
    inputSignal=cell2mat(t_sound_rir_conv_source');
    
    
    % %% 2. LPF 설계 (fir1 사용)
    % % 모든 필터는 동일한 차수 N을 가져야 위상 지연이 일치하여 PR이 가능함
    % lpf = cell(1, length(fc));
    % for i = 1:length(fc)
    %     lpf{i} = fir1(N, fc(i)/(fs/2));
    % end
    % 
    %% 3. 차분을 통한 서브밴드 필터(BPF) 생성
    filters = cell(1, length(fc)+1);%1x7의 밴드패스신호 생성
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
    
    %% 4. 신호 적용 및 완전 재구성 확인
    % inputSignal: [Length x 8ch] 마이크 신호라 가정
    band_signals = cell(1, length(fc)+1);
    for b = 1:7
        band_signals{b} = filter(filters{b},1, inputSignal);%(분자, 분모(FIR)계수, 필터링할 신호)
    end
    
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