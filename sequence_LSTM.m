
num_bands = length(fc)+1;
% 7개 네트워크를 위한 최상위 셀 생성 (net_1 ~ net_7 용도)
train_subbands = cell(num_bands, 1); 
version=1;
N_step = 1.5:0.5:3;  
deg_in = -90:10:90; % 10도 간격의 스캔 각도

bc=128;%미니배치
for N=1.5
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
                a = train_subbands{band_idx}{1, angle_idx}; 
                
                weight = (1/N)^(abs(deg_in(angle_idx))/10);% y = (1/N)^(|deg|/10)
        
                % 8채널 중 '센터 마이크'의 신호만 뽑아서 가중치 적용 -> [N x 1]
                y = a(:, center_mic) * weight; 
                
                % 세로로 데이터 누적 (시간 축으로 길게 이어붙임)
                X_train_concat = [X_train_concat; a]; 
                Y_train_concat = [Y_train_concat; y]; 
            end  
        % [중요] 매트랩 Sequence 학습을 위해 차원 뒤집기
        % X: [총 샘플수 x 8] -> [8 x 총 샘플수]
        % Y: [총 샘플수 x 1] -> [1 x 총 샘플수]
    % 밴드별 독립 맥스치 정규화
        band_max = max(abs(X_train_concat), [], 'all'); 
        if band_max == 0; band_max = 1; end 
        
        % 완벽하게 정규화된 고유 행렬 생성
        X_normalized = X_train_concat / band_max; 
        Y_normalized = Y_train_concat / band_max;
        
        % 훈련/검증 분리 (차원 뒤집기 전 2차원 상태에서 분리)
        val_idx = floor(size(X_normalized, 1) * 0.9);
        
    X_train_pure = X_normalized(1:val_idx-1, :)'; % [8 x 훈련샘플수]
    Y_train_pure = Y_normalized(1:val_idx-1, :)'; % [1 x 훈련샘플수]
    
    X_val_pure = X_normalized(val_idx:end, :)';   % [8 x 검증샘플수]
    Y_val_pure = Y_normalized(val_idx:end, :)';   % [1 x 검증샘플수]
    
% --- [훈련 데이터 쪼개기] ---
    chunkSize = 128; % 앞서 타협한 128샘플 타임스텝
    numChunks_train = floor(size(X_train_pure, 2) / chunkSize);
    
    X_cell = mat2cell(X_train_pure(:, 1:numChunks_train*chunkSize), 8, chunkSize*ones(1, numChunks_train))';
    Y_matrix = Y_train_pure(:, 1:numChunks_train*chunkSize); 
    Y_cell = Y_matrix(chunkSize:chunkSize:end)'; % [numChunks_train x 1] 실수 벡터
    
    % --- [검증 데이터 쪼개기 및 완전 정규화] ---
    numChunks_val = floor(size(X_val_pure, 2) / chunkSize);
    if numChunks_val == 0
        X_val_cell = {X_val_pure(:, 1:chunkSize)}; % 최소 1개 청크 크기 강제 확보
        Y_val_cell = Y_val_pure(chunkSize);
    else
        X_val_cell = mat2cell(X_val_pure(:, 1:numChunks_val*chunkSize), 8, chunkSize*ones(1, numChunks_val))';
        Y_val_matrix = Y_val_pure(:, 1:numChunks_val*chunkSize);
        Y_val_cell = Y_val_matrix(chunkSize:chunkSize:end)'; % [numChunks_val x 1] 실수 벡터
    end
    
    % --- 🚨 [핵심 해결] ValidationFrequency 및 배치 설정 수정 ---
    % 미니배치당 이터레이션 회수를 고려하여 에포크당 정확히 1회 검증이 수행되도록 주기를 계산합니다.
    iterationsPerEpoch = floor(numChunks_train / bc);
    valFrequency = max(1, iterationsPerEpoch); % 1에포크마다 검증 수행
    
    learning = 50;
    maxEpochs = learning;     
    inputSize = num_mics;      
    outputSize = 256;  
    numResponses = 1;   
    
    options = trainingOptions("adam", ... 
        InitialLearnRate = 0.001, ... 
        MaxEpochs = maxEpochs, ...
        miniBatchSize = bc, ... % 128개의 청크를 배치로 병렬 학습
        GradientThreshold = 1, ...
        Plots="training-progress", ...
        Shuffle = 'every-epoch', ...
        ExecutionEnvironment = "auto", ...
        ValidationData={X_val_cell, Y_val_cell}, ... % 차원이 완벽히 매칭된 검증셋
        ValidationFrequency = valFrequency, ...      % 에포크마다 꼬박꼬박 검증 로스 플로팅
        ValidationPatience = 10,  ...                % 로컬 미니마 탈출을 위해 10으로 완화
        OutputNetwork = 'best-validation-loss', ... 
        Verbose=0);


        layers = [
            sequenceInputLayer(8)
            lstmLayer(256, 'OutputMode', 'last') % ◀ 과거 100샘플의 위상 흐름을 기억 장치에 누적!
            leakyReluLayer(0.01)
            fullyConnectedLayer(1, Bias = 0, BiasLearnRateFactor = 0)
            regressionLayer
        ];
   
        % 각 밴드별 독립 네트워크 학습 후 셀에 저장
        % networks{band_idx} = trainNetwork(X_train_final, Y_train_final, layers, options);
        networks{band_idx} = trainNetwork(X_cell, Y_cell, layers, options);
        global_max{band_idx,1}=band_max;
    end
    disp('✅ 7개 밴드 네트워크 학습 모두 완료!');
    
    file_name=['sequence_LSTM_NNBF_learningdata_N',num2str(N),'version_',num2str(version),'.mat'];
    save(file_name,'networks','global_max')
end
%%
%leakageRelulayer를 없애고 Relu로 다시 해봤을떄 어떻게나오는지 확인하기