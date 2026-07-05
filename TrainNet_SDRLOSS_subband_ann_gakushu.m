
num_bands = length(fc)+1;
% 7개 네트워크를 위한 최상위 셀 생성 (net_1 ~ net_7 용도)
train_subbands = cell(num_bands, 1); 
deg_in = -90:10:90; % 10도 간격의 스캔 각도
N_step = 2.5:0.25:3;            % 논문에서 x제안한 최적의 지향성 계수 N (1.5, 2, 3 등 사용)
version=14;
anycomment='sdr+0.1rmse+softfilt+clipping+동적계수'
bc=2048*4;

        
learning = 100;
maxEpochs = learning;     %エポック数（学習回数）                      <------ 学習回数
        
inputSize = num_mics;      %入力数
outputSize = 256;  %中間層のユニット数                       <-------隠れユニット数 
numResponses = 1;   %全結合層の出力層
        
VF=10/5;
VP=50/5;
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
    
    % 2.밴드맥스 저장행렬
    global_max =cell(num_bands,1);
    
    % 전체 데이터를 관통하는 가장 큰 값으로 글로벌 정규화 (루프 밖에서 1회만 수행)
    % 이미 이전 코드에서 global_max를 구했다고 가정합니다.
    % train_subbands = train_subbands / global_max; 
    
    num_bands = 7;
    center_mic = 5; % 기준이 되는 센터 마이크 인덱스 (논문 기준)
    networks = cell(num_bands, 1); % 모델 7개를 저장할 방
    
    for band_idx = 1:num_bands
        fprintf('==== %d번 밴드 네트워크 데이터 구성 및 학습 ====\n', band_idx);
        chunkSize = bc; % 미니배치 크기 2048 고정

        % [중요] 밴드가 바뀔 때마다 학습 데이터 초기화
        X_train_concat = []; 
        Y_train_concat = []; 
        
        for angle_idx = 1:19
            % a는 [N(샘플수) x 8(채널)] 행렬
            a = train_subbands{band_idx}{1, angle_idx}; 
            
            weight = (1/N)^(abs(deg_in(angle_idx))/10);% y = (1/N)^(|deg|/10)
    
            % 8채널 중 '센터 마이크'의 신호만 뽑아서 가중치 적용 -> [N x 1]
            y = a(:, center_mic) * weight; 
            
            numChunksAngle = floor(size(a, 1) / chunkSize);
            a_trimmed = a(1:numChunksAngle * chunkSize, :);
            y_trimmed = y(1:numChunksAngle * chunkSize, :);
            
            % 배스크 크기에 딱 맞춰진 데이터만 누적 (각도 간 경계면 혼합 원천 차단)
            X_train_concat = [X_train_concat; a_trimmed]; 
            Y_train_concat = [Y_train_concat; y_trimmed];
        end
        
        % [중요] 매트랩 Sequence 학습을 위해 차원 뒤집기
        % X: [총 샘플수 x 8] -> [8 x 총 샘플수]
        % Y: [총 샘플수 x 1] -> [1 x 총 샘플수]
        % band_max = max(abs(X_train_concat), [], 'all'); 
        % if band_max == 0; band_max = 1; end % 0 나누기 방지
        band_max = sqrt(mean(X_train_concat.^2, 'all')) + 10^-8;
        if band_max == 0; band_max = 1; end % 0 나누기 방지

        % 1. 자투리를 버리고 전체 데이터를 정확히 chunkSize 배수로 클리핑 (Trim)
        % 여기서는 아직 정규화 전인 원본 X_train_concat을 기준으로 계산합니다.
        numChunksTotal = floor(size(X_train_concat, 1) / chunkSize);
        X_trimmed = X_train_concat(1:numChunksTotal*chunkSize, :);
        Y_trimmed = Y_train_concat(1:numChunksTotal*chunkSize, :);
        
        % 2. 밸리데이션 생성 및 믹싱 (8:2 셔플)
        rng(42); % 실험 재현성 시드 고정
        shuffledChunkIdx = randperm(numChunksTotal);
        
        valStartChunk = floor(numChunksTotal * 0.8) + 1;
        train_chunks = shuffledChunkIdx(1:valStartChunk-1);
        val_chunks = shuffledChunkIdx(valStartChunk:end);
        
        % 3. 데이터를 옮겨 담을 빈 방(Zeros 행렬) 미리 만들기
        num_train_samples = length(train_chunks) * chunkSize;
        num_val_samples = length(val_chunks) * chunkSize;
        
        X_train_new = zeros(num_train_samples, inputSize);
        Y_train_new = zeros(num_train_samples, 1);
        
        X_val_new = zeros(num_val_samples, inputSize);
        Y_val_new = zeros(num_val_samples, 1);

        % =========================================================
        % 4. 훈련 데이터 복사 (For 문으로 행 단위 통째 복사)
        % =========================================================
        for i = 1:length(train_chunks)
            src_idx = train_chunks(i);
            src_start = (src_idx - 1) * chunkSize + 1;
            src_end   = src_idx * chunkSize;
            
            dst_start = (i - 1) * chunkSize + 1;
            dst_end   = i * chunkSize;
            
            X_train_new(dst_start:dst_end, :) = X_trimmed(src_start:src_end, :);
            Y_train_new(dst_start:dst_end, :) = Y_trimmed(src_start:src_end, :);
        end
        
        % =========================================================
        % 5. 검증 데이터 복사 (동일한 원리)
        % =========================================================
        for i = 1:length(val_chunks)
            src_idx = val_chunks(i);
            src_start = (src_idx - 1) * chunkSize + 1;
            src_end   = src_idx * chunkSize;
            
            dst_start = (i - 1) * chunkSize + 1;
            dst_end   = i * chunkSize;
            
            X_val_new(dst_start:dst_end, :) = X_trimmed(src_start:src_end, :);
            Y_val_new(dst_start:dst_end, :) = Y_trimmed(src_start:src_end, :);
        end
        
        % =========================================================
        % 6. 최종 정규화 적용 (가장 마지막에 일괄 적용) 및 변수명 덮어쓰기
        % =========================================================
        X_train_final = X_train_new / band_max;
        Y_train_final = Y_train_new / band_max;
        X_val = X_val_new / band_max;
        Y_val = Y_val_new / band_max;
        % =========================================================================
        % [수정본] 직관적인 원본 데이터를 7개의 네트워크 학습용 구조로 재배치
        % =========================================================================

        % === Ttrainnet ===%
            options = trainingOptions("adam", ... 
            InitialLearnRate = 0.001, ... % 正規化されているので高めでもOK
            MaxEpochs = maxEpochs, ...
            miniBatchSize = bc, ...    
            GradientThreshold = Inf, ...
            Plots="training-progress", ...
            Shuffle = 'every-epoch', ...
            ExecutionEnvironment = "auto", ...
            ValidationData={X_val, Y_val}, ... % 검증 데이터 지정
            ValidationFrequency = floor((size(X_train_final, 1)/bc)/VF), ... % 몇 번의 이터레이션마다 검증할지 (데이터 크기에 맞게 조절)
            ValidationPatience = VP,  ...          % 검증 Loss가 5회 연속 안 떨어지면 조기 종료 (Early Stopping)
            OutputNetwork = 'best-validation-loss', ... % 검증 로스가 최소가 되는지점의 값을 반환
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
        ];
        
        % options 설정 (미리 외부에 정의해둔 options 사용)
        % options = ... 
        
        % 각 밴드별 독립 네트워크 학습 후 셀에 저장
        networks{band_idx} = trainnet(X_train_final, Y_train_final, layers,@sisdrLossLayer, options);
        % networks{band_idx} = trainNetwork(X_cell, Y_cell, layers, options);
        global_max{band_idx,1}=band_max;
    end
    
    disp('✅ 7개 밴드 네트워크 학습 모두 완료!');
    file_name=[anycomment,'customizedLoss_NNBF_learningdata_subandsoft_N',num2str(N),'version_',num2str(version),'.mat'];
    save(file_name,'networks','global_max')
    %%
    %global max를 없애고, 각 밴드별 max값으로 구하면 괜찮지않냐
    %6번 7번밴드만 RMSE천천히 줄어드는게 이유가 있지않을까... 주파수대역이 넓어서 정보도 많나?
    %leakageRelulayer를 없애고 Relu로 다시 해봤을떄 어떻게나오는지 확인하기
end

