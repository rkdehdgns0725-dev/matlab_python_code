function [SI_SDR, SI_SIR, SI_SAR, e_targetpow, e_interfpow, e_artifpow] = SI_metrics(s, hat_s, interfer)

    % 列ベクトル化（保険）
    s        = s(:);
    hat_s   = hat_s(:);
    interfer = interfer(:);

    %% === 1. Target projection (SI-SDR) ===
    alpha = (hat_s' * s) / (s' * s);
    e_target = alpha * s;
    %% === 2. Residual ===
    e_res = hat_s - e_target;

    %% === 3. e_interf : projection onto span{s, n} ===
    % 小行列（2x2）で解く
    C = [s' * s,        s' * interfer;
         interfer' * s, interfer' * interfer];

    d = [s' * e_res;
         interfer' * e_res];

    coeff = pinv(C) * d;     % 2x1
    e_interf = s * coeff(1) + interfer * coeff(2);

    %% === 4. Artifact ===
    e_artif = e_res - e_interf;

    %% === 5. Metrics === %%
    SI_SDR = 10 * log10( sum(e_target.^2) / sum(e_res.^2) );
    SI_SIR = 10 * log10( sum(e_target.^2) / sum(e_interf.^2) );
    SI_SAR = 10 * log10( sum(e_target.^2) / sum(e_artif.^2) );
    e_targetpow = sum(e_target.^2);
    e_interfpow = sum(e_interf.^2);
    e_artifpow = sum(e_artif.^2);
end
