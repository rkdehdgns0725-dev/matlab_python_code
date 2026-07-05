
num_bands = length(fc)+1;
% 7개 네트워크를 위한 최상위 셀 생성 (net_1 ~ net_7 용도)
train_subbands = cell(num_bands, 1); 
deg_in = -90:10:90; % 10도 간격의 스캔 각도
N_step = 1.5:0.5:3;            % 논문에서 x제안한 최적의 지향성 계수 N (1.5, 2, 3 등 사용)
version=001004;
anycomment='sdr+rmse+softfilt'
        
        
learning = 100;
maxEpochs = learning;     %エポック数（学習回数）                      <------ 学習回数
        
inputSize = num_mics;      %入力数
outputSize = 256;  %中間層のユニット数                       <-------隠れユニット数 
numResponses = 1;   %全結合層の出力層
        
VF=4;

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
        band_max = max(abs(X_train_concat), [], 'all'); 
        if band_max == 0; band_max = 1; end % 0 나누기 방지
        
        X_train_final = X_train_concat / band_max; 
        Y_train_final = Y_train_concat / band_max;
  
        chunkSize = bc; % 미니배치 크기 2048 고정
        
        % 1. 자투리 샘플을 버리고 전체 데이터를 정확히 2048의 배수로 클리핑
        numChunksTotal = floor(size(X_train_final, 1) / chunkSize);
        X_trimmed = X_train_final(1:numChunksTotal*chunkSize, :);
        Y_trimmed = Y_train_final(1:numChunksTotal*chunkSize, :);
        
        % 2. 3차원 텐서로 변환하여 2048개 샘플씩 연속성을 보존한 채 묶음
        X_reshape = reshape(X_trimmed, chunkSize, inputSize, numChunksTotal);
        Y_reshape = reshape(Y_trimmed, chunkSize, 1, numChunksTotal);
        
        % 3. 🚨 [핵심] 청크 조각들의 순서를 무작위로 셔플하여 전 각도(-90~90도)를 뒤섞음
        rng(42); % 실험 재현성 시드 고정
        shuffledChunkIdx = randperm(numChunksTotal);
        X_reshape = X_reshape(:, :, shuffledChunkIdx);
        Y_reshape = Y_reshape(:, :, shuffledChunkIdx);
        
        % 4. 9:1 비율로 청크 개수를 분할 계산
        valStartChunk = floor(numChunksTotal * 0.9) + 1;
        
        % 90%는 훈련 청크 방에 격납
        X_train_chunk = X_reshape(:, :, 1:valStartChunk-1);
        Y_train_chunk = Y_reshape(:, :, 1:valStartChunk-1);
        
        % 10%는 검증 청크 방에 격납 (이제 모든 각도가 이 안에 골고루 섞임!)
        X_val_chunk = X_reshape(:, :, valStartChunk:end);
        Y_val_chunk = Y_reshape(:, :, valStartChunk:end);
        
        % 5. trainNetwork가 정상 공급받을 수 있도록 각각 2D 행렬 형태로 최종 복원
        X_train_final = reshape(X_train_chunk, [], inputSize);
        Y_train_final = reshape(Y_train_chunk, [], 1);
        
        X_val = reshape(X_val_chunk, [], inputSize);
        Y_val = reshape(Y_val_chunk, [], 1);
        % =========================================================================
        % [수정본] 직관적인 원본 데이터를 7개의 네트워크 학습용 구조로 재배치
        % =========================================================================
        fprintf("Band %d RMS = %.6f\n", ...
        band_idx, rms(X_train_final(:)));
        fprintf("Band %d Target RMS = %.6f\n", ...
        band_idx, rms(Y_train_final(:)));
        fprintf("Band %d Target Peak = %.6f\n", ...
        band_idx, max(abs(Y_train_final(:))));
    end
end

