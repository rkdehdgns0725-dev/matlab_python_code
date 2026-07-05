function loss = sisdrLossLayer(Y, T)
    persistent cnt
    if isempty(cnt)
        cnt = 0;
    end
    cnt = cnt + 1;
    
    eps_val = 1e-8;
    clipping_th = 0; % dB

    % 💡 핵심 수정: 'all'을 사용하여 데이터가 [1x2048]이든 [2048x1]이든 
    % 전체를 하나의 오디오 블록으로 통째로 병합합니다.
    Y_zero = Y - mean(Y, 'all');
    T_zero = T - mean(T, 'all');
    
    dot_product = sum(Y_zero .* T_zero, 'all')+eps_val;
    t_energy    = sum(T_zero .^ 2, 'all') + eps_val;
    alpha       = dot_product ./ t_energy;

    s_target_SISDR    = alpha.*T_zero;
    s_target    = T_zero;
    e_res       = Y_zero - s_target;
    

    % 'all'을 통해 에너지를 완전히 합산하여 완벽한 1x1 스칼라로 만듭니다.
    target_power = sum(s_target .^ 2, 'all') + eps_val;
    noise_power  = sum(e_res .^ 2, 'all') + eps_val;
    target_power_SISDR = sum(s_target_SISDR .^ 2, 'all') + eps_val;

    % SDR 및 트랩 방지 계산
    sdr = 10 * log10(target_power / noise_power);
    SI_SDR = 10 * log10(target_power_SISDR / noise_power);    
    % distance = max(si_sdr, -15)-clipping_th; 
    distance = sdr-clipping_th; 
    
    alpha_weight = 100.0;  
    beta_decay   = 0.05; 
    dynamic_mse_weight = max(1.0, alpha_weight * exp(-beta_decay * distance));
    
    mse_value = mean((Y-T).^2, 'all');
    
    % 모든 항이 스칼라이므로 연산 후 완벽한 1x1 결과 도출
    loss_labeled = -sdr + (dynamic_mse_weight .* mse_value);
    % loss_labeled = -si_sdr ;
    loss = stripdims(loss_labeled);
    
    % 500 이터레이션마다 정보 출력
    if mod(cnt, 500) == 0
        fprintf("SDR=%6.2f dB |SI-SDR=%6.2f dB | MSE=%.5f | Weight=%6.2f | Loss=%6.2f | TarPow=%.5f | NoiPow=%.5f\n",...
            gather(extractdata(sdr)),...
            gather(extractdata(SI_SDR)),...
            gather(extractdata(mse_value)),...
            gather(extractdata(dynamic_mse_weight)),...
            gather(extractdata(loss)),...
            gather(extractdata(target_power)),...
            gather(extractdata(noise_power)));
        
    end
end



% function loss = sisdrLossLayer(Y, T)
%     persistent cnt
%     if isempty(cnt)
%         cnt = 0;
%     end
%     cnt = cnt + 1;
% 
%     eps_val = 1e-8;
%     clipping_th = -80; % dB
%     working_dim = 2;   % trainnet의 기본 배치 차원은 1번 축(세로)
% 
%     % 바이어스 제거
%     % Y_zero = Y - mean(Y, working_dim);
%     % T_zero = T - mean(T, working_dim);
%     Y_zero = Y ;
%     T_zero = T;
% 
%     % Target Projection
%     dot_product = sum(Y_zero .* T_zero, working_dim);
%     t_energy    = sum(T_zero .^ 2, working_dim) + eps_val;
%     alpha       = dot_product ./ t_energy;
% 
%     % s_target    = alpha.*T_zero;
%     s_target    = T_zero;
%     e_res       = Y_zero - s_target;
% 
%     target_power = sum(s_target .^ 2, working_dim) + eps_val;
%     noise_power  = sum(e_res .^ 2, working_dim) + eps_val;
% 
%     % SI-SDR 계산 및 클리핑
%     si_sdr = 10 * log10(target_power ./ noise_power);
%     si_sdr = max(si_sdr, clipping_th);
% 
%     distance =  max(si_sdr,0); 
% 
%     % 가중치 동적 계산
%     alpha_weight = 1000.0;  
%     beta_decay   = 0.05; 
%     dynamic_mse_weight = max(1.0, alpha_weight * exp(-beta_decay * distance));
%     mse_value=mean((Y-T).^2,'all');
%     % 최종 손실 (모든 배치 샘플에 대한 평균 스칼라 값 반환)
%     loss = -mean(distance,'all') + dynamic_mse_weight .* mean((Y-T).^2);%sisdr대신 0dB이하일때는 
% 
%     % 500 이터레이션마다 정보 출력 (타깃 파워, 노이즈 파워 추가)
%     if mod(cnt, 500) == 0
%         fprintf("SI-SDR=%6.2f dB | MSE=%.5f | Weight=%6.2f | Loss=%6.2f | TarPow=%.5f | NoiPow=%.5f\n",...
%             gather(mean(extractdata(si_sdr), 'all')),...
%             gather(mean(extractdata(mse_value), 'all')),...
%             gather(mean(extractdata(dynamic_mse_weight), 'all')),...
%             gather(mean(extractdata(loss), 'all')),...
%             gather(mean(extractdata(target_power), 'all')),...
%             gather(mean(extractdata(noise_power), 'all')));
%     end
% end