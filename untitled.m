%% Sub-band Beamformer Configuration via Simulated Annealing
% 2026-06-06 組み合わせ最適化問題課題
% Objective: Find the optimal V = [x1, x2, s1, y1] to minimize H(V)

clear; clc;
result=zeros(1,100);
%% 1. パラメータおよび制約条件の設定 (Hyperparameters)
d = [4; 2];       % 各サブバンドの単独音質改善利得 (SI-SDR改善量)
w12 = 3;          % 隣接バンド間の位相歪みペナルティ
b = [5; 3];       % 各バンドの雑音抑圧貢献度 (SNR改善量)
c = [2; 2];       % 各バンドの計算コスト (処理時間/演算量)

B_target = 4;     % 最小雑音抑圧目標値 (下限値)
C_max = 3;        % 最大計算資源限界値 (上限値)

lambda = 10;      % 雑音抑圧制約項の重み係数 (ペナルティ強度)
gamma = 10;       % 計算資源制約項の重み係数 (ペナルティ強度)

%% 2. アニーリングアルゴリズムのパラメータ設定 (SA Parameters)
T_init = 500.0;   % 初期温度
T_min = 0.001;    % 終了温度 (凍結基準)
alpha = 0.95;     % 冷却係数 (Cooling rate)
max_unfreeze = 100; % 温度を更新する総ステップ数 (外側ループの回数)
max_iter = 50;    % 各温度における遷移試行回数 (内側ループによる熱平衡状態の確保)

% 初期状態のランダム生成 (V = [x1, x2, s1, y1])
current_V = randi([0, 1], 4, 1);
current_E = calculate_hamiltonian(current_V, d, w12, b, c, B_target, C_max, lambda, gamma);

best_V = current_V;
best_E = current_E;

T = T_init;
history_E = [];   % 収束特性可視化のためのエネルギー履歴記録配列

%% 3. シミュレーティド・アニーリング（二重ループ）の実行
%  % 結果の再現性を担保するための乱数シード固定
for repeat=1:1000
    for m = 1:max_unfreeze
        if T < T_min; break; end
        
        for n = 1:max_iter
            % 近傍状態の生成 (4ビットのうちランダムに1ビットを反転)
            next_V = current_V;
            flip_idx = randi(4);
            next_V(flip_idx) = 1 - next_V(flip_idx);
                
            % 近傍状態におけるハミルトニアン（エネルギー）の計算
            next_E = calculate_hamiltonian(next_V, d, w12, b, c, B_target, C_max, lambda, gamma);
             
            % エネルギー変化量の計算
            dE = next_E - current_E;
              
            % メトロポリス基準 (Metropolis Criterion) に基づく状態遷移確率の決定
            if dE < 0
                % エネルギーが減少した場合は無条件で状態を更新
                current_V = next_V;
                current_E = next_E;
            else
                % エネルギーが増加した場合でも確率的に受け入れ (局所解からの脱出メカニズム)
                p = exp(-dE / T);
                if rand() < p
                    current_V = next_V;
                    current_E = next_E;
                end
            end
            
            % 過去最小エネルギー状態（最良解）の更新チェック
            if current_E < best_E
                best_V = current_V;
                best_E = current_E;
            end
        end
        
        % 内側ループによる熱平衡の達成後、温度を冷却しエネルギを記録
        history_E = [history_E, best_E];
        T = T * alpha;
    end
    result(repeat)=best_E;
end
%% 4. 最適化結果の出力および収束特性の可視化
fprintf('===============================================\n');
fprintf('  Simulated Annealing Optimization Result\n');
fprintf('===============================================\n');
fprintf('Optimal State Vector V* = [%d, %d, %d, %d]\n', best_V(1), best_V(2), best_V(3), best_V(4));
fprintf(' -> Sub-band 1 Mask (x1) : %d (1: NNBF, 0: DS)\n', best_V(1));
fprintf(' -> Sub-band 2 Mask (x2) : %d (1: NNBF, 0: DS)\n', best_V(2));
fprintf(' -> Noise Slack    (s1) : %d\n', best_V(3));
fprintf(' -> Resource Slack (y1) : %d\n', best_V(4));
fprintf('Minimum Energy H(V*)    : %d\n', best_E);
fprintf('===============================================\n');

% エネルギー収束曲線の描画
figure;
plot(history_E, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
grid on;
title('Simulated Annealing Energy Convergence');
xlabel('Temperature Cooling Steps');
ylabel('Minimum Energy H(V)');

%% 5. ハミルトニアン（エネルギー関数）計算サブ関数
function H = calculate_hamiltonian(V, d, w12, b, c, B_target, C_max, lambda, gamma)
    % ベクトルの要素分解
    x1 = V(1); x2 = V(2); s1 = V(3); y1 = V(4);
    
    % 各評価項のエネルギー計算
    E_gain = - (d(1)*x1 + d(2)*x2);
    E_penalty = w12 * (x1 - x2)^2;
    
    % 不等式制約条件をスラック変数を用いて変換したペナルティ項
    E_noise = lambda * (B_target + s1 - (b(1)*x1 + b(2)*x2))^2;
    E_resource = gamma * (c(1)*x1 + c(2)*x2 + y1 - C_max)^2;
    
    % 総ハミルトニアンの算出
    H = E_gain + E_penalty + E_noise + E_resource;
end