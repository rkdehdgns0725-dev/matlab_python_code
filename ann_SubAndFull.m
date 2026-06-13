
num_bands = length(fc)+1;
% 7개 네트워크를 위한 최상위 셀 생성 (net_1 ~ net_7 용도)
train_subbands = cell(num_bands, 1); 

for b = 1:num_bands
    % 각 밴드별로 19개 각도의 [N x 8] 행렬을 담을 임시 저장소
    angle_cell = cell(1, num_sources); 

    for angle_idx = 1:num_sources
        % 1. 데이터 길이 파악 (1번 마이크, 현재 각도, 현재 밴드 기준)
        % 셀 접근 방식 주의: bandpassed_dataset{채널, 각도}{밴드}
        sig_len = length(bandpassed_dataset{1, angle_idx}{b});

        % 2. 현재 각도(angle_idx)에서 8개 마이크의 신호를 담을 [N x 8] 빈 행렬
        ch_matrix = zeros(sig_len, mic_ch);

        % 3. 8개 마이크 채널을 돌면서 빈 행렬 채우기
        for ch_idx = 1:mic_ch
            % 무조건 열 벡터(:) 형태로 가져와서 행렬의 각 열에 꽂아 넣음
            ch_matrix(:, ch_idx) = bandpassed_dataset{ch_idx, angle_idx}{b}(:);
        end

        % 4. 완성된 [N x 8] 8채널 데이터를 현재 각도의 방에 격납
        angle_cell{1, angle_idx} = ch_matrix;%1x19 cell에는 Nx8데이터
    end

    % 5. 19개 각도의 데이터가 모두 모이면, 이를 b번째 밴드 훈련소에 최종 격납
    train_subbands{b} = angle_cell;% 7x1bandpassed신호
end

fprintf('데이터 재배치 완료: %d개 밴드에 대해 각각 %d개 각도의 8채널 데이터 생성 완료!\n', num_bands, num_sources);

all_data_matrix = cell2mat([train_subbands{:}]); 

% 2. 이제 순수한 숫자형 행렬이 되었으므로 'all' 옵션으로 완벽한 글로벌 최댓값을 찾습니다.
global_max =cell(num_bands,1);

% 전체 데이터를 관통하는 가장 큰 값으로 글로벌 정규화 (루프 밖에서 1회만 수행)
% 이미 이전 코드에서 global_max를 구했다고 가정합니다.
% train_subbands = train_subbands / global_max; 

num_bands = 7;
center_mic = 5; % 기준이 되는 센터 마이크 인덱스 (논문 기준)
networks = cell(num_bands, 1); % 모델 7개를 저장할 방

for band_idx = 1:num_bands
    fprintf('==== %d번 밴드 네트워크 데이터 구성 및 학습 ====\n', band_idx);
    
    % [중요] 밴드가 바뀔 때마다 학습 데이터 초기화
    X_train_concat = []; 
    Y_train_concat = []; 
    for angle_idx = 1:19
        % a는 [N(샘플수) x 8(채널)] 행렬
        %a = train_subbands{band_idx}{1, angle_idx}; 
        fullband = t_sound_rir_conv_source_8_18(:,angle_idx);
        fullband=cell2mat(fullband');
        % ----------------------------------------------------
        % [핵심] 교사 데이터(Target) Y 생성 로직 (나카지마 방식)
        % ----------------------------------------------------
        if angle_idx == 10 % 0도 (정면 타겟 방향)
            weight = 1.0; 
        else               % 타겟 이외의 방향 (노이즈)
            weight = 0.0001; % 0 데이터 학습일 경우
            % weight = 10^(-20/20); % 만약 특정 감쇄(예: -20dB)를 줄 경우
        end
        
        % 8채널 중 '센터 마이크'의 신호만 뽑아서 가중치 적용 -> [N x 1]
        y = fullband(:, center_mic) * weight; 
        
        % 세로로 데이터 누적 (시간 축으로 길게 이어붙임)
        X_train_concat = [X_train_concat; fullband]; 
        Y_train_concat = [Y_train_concat; y]; 
    end
    
    for angle_idx = 1:19
        % a는 [N(샘플수) x 8(채널)] 행렬
        a = train_subbands{band_idx}{1, angle_idx}; 
        
        % ----------------------------------------------------
        % [핵심] 교사 데이터(Target) Y 생성 로직 (나카지마 방식)
        % ----------------------------------------------------
        if angle_idx == 10 % 0도 (정면 타겟 방향)
            weight = 1.0; 
        else               % 타겟 이외의 방향 (노이즈)
            weight = 0.0001; % -80dB
            % weight = 10^(-20/20); % 만약 특정 감쇄(예: -20dB)를 줄 경우
        end
        
        % 8채널 중 '센터 마이크'의 신호만 뽑아서 가중치 적용 -> [N x 1]
        y = a(:, center_mic) * weight; 
        
        % 세로로 데이터 누적 (시간 축으로 길게 이어붙임)
        X_train_concat = [X_train_concat; a]; 
        Y_train_concat = [Y_train_concat; y]; 
    end
    
    % [중요] 매트랩 Sequence 학습을 위해 차원 뒤집기
    % X: [총 샘플수 x 8] -> [8 x 총 샘플수]
    % Y: [총 샘플수 x 1] -> [1 x 총 샘플수]
    band_max = max(abs(X_train_concat), [], 'all'); 
    if band_max == 0; band_max = 1; end % 0 나누기 방지
    
    X_train_final = X_train_concat / band_max; 
    Y_train_final = Y_train_concat / band_max;



    val_idx = floor(size(X_train_final, 1) * 0.9);
    
    X_val = X_train_final(val_idx:end, :);
    Y_val = Y_train_final(val_idx:end, :);
    X_train_final = X_train_final(1:val_idx-1,:);
    Y_train_final = Y_train_final(1:val_idx-1,:);

    % chunkSize = 100; % 미니배치 사이즈와 맞추면 좋습니다
    % 
    % % 데이터를 [8 x 총샘플수]에서 [8 x chunkSize] 조각들로 쪼갭니다.
    % % mat2cell은 지정된 길이로 행렬을 싹둑 자릅니다.
    % numChunks = floor(size(X_train_final, 2) / chunkSize);
    % X_cell = mat2cell(X_train_final(:, 1:numChunks*chunkSize), 8, chunkSize*ones(1, numChunks))';
    % Y_cell = mat2cell(Y_train_final(:, 1:numChunks*chunkSize), 1, chunkSize*ones(1, numChunks))';
    % % ----------------------------------------------------
    % 네트워크 레이어 설정 및 학습
    % ----------------------------------------------------
    % =========================================================================
    % [수정본] 직관적인 원본 데이터를 7개의 네트워크 학습용 구조로 재배치
    % =========================================================================
    
    
    learning = 30;
    maxEpochs = learning;     %エポック数（学習回数）                      <------ 学習回数
    maxEpochs2 = learning;
    
    inputSize = num_mics;      %入力数
    outputSize = 256;  %中間層のユニット数                       <-------隠れユニット数 
    numResponses = 1;   %全結合層の出力層
    
    bc=2048;
    VF=4;
    % === Ttrainnet ===%
    options = trainingOptions("adam", ... 
        InitialLearnRate = 0.001, ... % 正規化されているので高めでもOK
        MaxEpochs = maxEpochs, ...
        miniBatchSize = bc, ...    
        GradientThreshold = 1, ...
        Plots="training-progress", ...
        Shuffle = 'every-epoch', ...
        ExecutionEnvironment = "auto", ...
        ValidationData={X_val, Y_val}, ... % 검증 데이터 지정
        ValidationFrequency = floor((size(X_train_final, 1)/bc)/VF), ...      % 몇 번의 이터레이션마다 검증할지 (데이터 크기에 맞게 조절)
        ValidationPatience = 5,  ...          % 검증 Loss가 5회 연속 안 떨어지면 조기 종료 (Early Stopping)
        Verbose=0);


    layers = [
        featureInputLayer(inputSize)
        % sequenceInputLayer(inputSize)

        fullyConnectedLayer(outputSize)
        leakyReluLayer(0.01)
        % ReluLayer
    
        fullyConnectedLayer(outputSize/2)
        leakyReluLayer(0.01)
        % ReluLayer
    
        fullyConnectedLayer(numResponses, Bias = 0, BiasLearnRateFactor = 0)
        regressionLayer('Name', 'rmse_loss')
    ];
    
    % options 설정 (미리 외부에 정의해둔 options 사용)
    % options = ... 
    
    % 각 밴드별 독립 네트워크 학습 후 셀에 저장
    networks{band_idx} = trainNetwork(X_train_final, Y_train_final, layers, options);
    % networks{band_idx} = trainNetwork(X_cell, Y_cell, layers, options);
    global_max{band_idx,1}=band_max;
end

disp('✅ 7개 밴드 네트워크 학습 모두 완료!');
save('NNBF_learningdata_subband_v3','networks','global_max')
%%
%global max를 없애고, 각 밴드별 max값으로 구하면 괜찮지않냐
%6번 7번밴드만 RMSE천천히 줄어드는게 이유가 있지않을까... 주파수대역이 넓어서 정보도 많나?
%leakageRelulayer를 없애고 Relu로 다시 해봤을떄 어떻게나오는지 확인하기







