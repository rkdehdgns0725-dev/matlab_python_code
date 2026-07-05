function loss = sisdrLoss(Y, T)
    eps_val = 1e-8;
    error("sisdrLoss is called");
    Y_zero = Y - mean(Y, 1);
    T_zero = T - mean(T, 1);
    
    dot_product = sum(Y_zero .* T_zero, 1);
    
    % Target 에너지 0 방지
    t_energy = max(sum(T_zero .^ 2, 1), eps_val); 
    
    alpha    = dot_product ./ t_energy;
    e_target = alpha .* T_zero;
    % e_target = T_zero;

    % Residual 에너지 0 방지
    e_res = Y_zero - e_target;
    res_energy = max(sum(e_res.^2, 1), eps_val); 
    target_energy_adj = max(sum(e_target.^2, 1), eps_val);
    
    si_sdr = 10 * log10( target_energy_adj ./ res_energy );
    % 🚨 핵심 수정: SI-SDR이 과도한 음수로 떨어져 트랩에 빠지는 것을 방지 (-15dB 지점에서 클리핑)
    si_sdr = max(si_sdr, -80);
    distance = si_sdr + 80; 
            
    % 하이퍼파라미터 설정
    alpha_weight = 50.0;  % 트랩(-15dB)에 갇혔을 때 때릴 강력한 MSE 가중치 (기존 0.1에서 상향)
    beta_decay   = 0.15; % 양수로 갈수록 MSE 가중치를 얼마나 빠르게 감소시킬지 결정하는 감쇠율
            
    % 지수 감쇠 수식 적용
    dynamic_mse_weight = alpha_weight * exp(-beta_decay * distance);
            
    % 최종 손실 계산
    loss = -si_sdr + dynamic_mse_weight * mean((Y-T).^2, 'all');
end