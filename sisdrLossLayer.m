function loss = sisdrLossLayer(Y, T)
persistent cnt
    if isempty(cnt)
        cnt = 0;
    end
    cnt = cnt + 1;
    
    eps_val = 1e-8;
    clipping_th = 0; % 너무 떨어졌을 때 방어선 (dB)
    
    % 1. 시간 차원(보통 1번 축) 설정 
    working_dim = 2; 
    
    % 2. DC Offset 제거 (배치 내 각 샘플별로 독립적으로 수행)
    Y_zero = Y - mean(Y, working_dim);
    T_zero = T - mean(T, working_dim);
    
    % =======================================================
    % [모니터링용 일반 SDR 계산] (스케일 조정 없이 원본 그대로 비교)
    % =======================================================
    target_power_sdr = sum(T_zero .^ 2, working_dim) + eps_val;
    e_res_sdr        = Y_zero - T_zero;
    noise_power_sdr  = sum(e_res_sdr .^ 2, working_dim) + eps_val;
    
    sdr_batch = 10 * log10(target_power_sdr ./ noise_power_sdr);
    sdr_mean  = mean(sdr_batch, 'all'); % 화면 출력용
    
    % =======================================================
    % [학습 최적화용 SI-SDR 계산] (볼륨 스케일 영점 조절 후 비교)
    % =======================================================
    dot_product = sum(Y_zero .* T_zero, working_dim);
    alpha       = dot_product ./ target_power_sdr; % [1 x Batch] 크기의 스케일 팩터
    
    s_target_sisdr = alpha .* T_zero;         % 스케일이 맞춰진 타겟
    e_res_sisdr    = Y_zero - s_target_sisdr; % 잔여 노이즈 (오차)
    
    target_power_sisdr = sum(s_target_sisdr .^ 2, working_dim) + eps_val;
    noise_power_sisdr  = sum(e_res_sisdr .^ 2, working_dim) + eps_val;
    
    si_sdr_batch = 10 * log10(target_power_sisdr ./ noise_power_sisdr);
    si_sdr_batch = max(si_sdr_batch, clipping_th); 
    si_sdr_mean  = mean(si_sdr_batch, 'all'); % 로스 계산용
    
    % =======================================================
    % [MSE 계산 및 동적 가중치 적용]
    % =======================================================
    mse_batch = mean((Y - T).^2, working_dim);
    mse_mean  = mean(mse_batch, 'all');
    
    alpha_weight = 100.0;  
    beta_decay   = 0.1; 
    dynamic_mse_weight = max(1.0, alpha_weight * exp(-beta_decay * sdr_mean));%sisdr
    
    % 3. 최종 하이브리드 로스 (-SI_SDR 최소화 + MSE 페널티)
    loss = -sdr_mean + (dynamic_mse_weight .* mse_mean);%sisdr
    loss = stripdims(loss); 
    
    % 4. 500 이터레이션마다 정보 출력 (SDR 추가)
    if mod(cnt, 500) == 0
        fprintf("SDR=%6.2f dB | SI-SDR=%6.2f dB | MSE=%.5f | Weight=%6.2f | Loss=%6.2f\n",...
            gather(extractdata(sdr_mean)),...
            gather(extractdata(si_sdr_mean)),...
            gather(extractdata(mse_mean)),...
            gather(extractdata(dynamic_mse_weight)),...
            gather(extractdata(loss)));
    end
end
%     persistent cnt
%     if isempty(cnt)
%         cnt = 0;
%     end
%     cnt = cnt + 1;
% 
%     eps_val = 1e-8;
%     clipping_th = 0; % dB
%     working_dim=2;
%     % 💡 핵심 수정: 'all'을 사용하여 데이터가 [1x2048]이든 [2048x1]이든 
%     % 전체를 하나의 오디오 블록으로 통째로 병합합니다.
%     Y_zero = Y - mean(Y, working_dim);
%     T_zero = T - mean(T, working_dim);
% 
%     dot_product = sum(Y_zero .* T_zero, working_dim)+eps_val;
%     t_energy    = sum(T_zero .^ 2, working_dim) + eps_val;
%     alpha       = dot_product ./ t_energy;
% 
%     s_target_SISDR    = alpha.*T_zero;
%     s_target    = T_zero;
%     e_res       = Y_zero - s_target;
% 
% 
%     % 'all'을 통해 에너지를 완전히 합산하여 완벽한 1x1 스칼라로 만듭니다.
%     target_power = sum(s_target .^ 2, working_dim) + eps_val;%
%     noise_power  = sum(e_res .^ 2, working_dim) + eps_val;%
%     target_power_SISDR = sum(s_target_SISDR .^ 2, working_dim) + eps_val;
% 
%     % SDR 및 트랩 방지 계산
%     sdr = 10 * log10(target_power / noise_power);
%     SI_SDR = 10 * log10(target_power_SISDR / noise_power);    %
%     % distance = max(si_sdr, -15)-clipping_th; 
%     distance = sdr-clipping_th; 
% 
%     alpha_weight = 100.0;  
%     beta_decay   = 0.05; 
%     dynamic_mse_weight = max(1.0, alpha_weight * exp(-beta_decay * distance));
% 
%     mse_value = mean((Y-T).^2, 'all');
% 
%     % 모든 항이 스칼라이므로 연산 후 완벽한 1x1 결과 도출
%     loss_labeled = -SI_SDR + (dynamic_mse_weight .* mse_value);
%     loss = stripdims(loss_labeled);
% 
%     % 500 이터레이션마다 정보 출력
%     if mod(cnt, 500) == 0
%         fprintf("SDR=%6.2f dB |SI-SDR=%6.2f dB | MSE=%.5f | Weight=%6.2f | Loss=%6.2f | TarPow=%.5f | NoiPow=%.5f\n",...
%             gather(extractdata(sdr)),...
%             gather(extractdata(SI_SDR)),...
%             gather(extractdata(mse_value)),...
%             gather(extractdata(dynamic_mse_weight)),...
%             gather(extractdata(loss)),...
%             gather(extractdata(target_power)),...
%             gather(extractdata(noise_power)));
% 
%     end
% end
% 

