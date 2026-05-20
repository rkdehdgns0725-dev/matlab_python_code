% X_train: 입력 데이터, Y_train: 교사 데이터
X_train = []; Y_train = [];

for ang = 1:num_sources
    % 현재 각도의 8채널 데이터를 샘플 수 x 8 행렬로 변환
    current_input = cell2mat(noise_rir_conv_source(:, ang)'); % [N x 8]
    if ang == 10 % 0도(정중앙)인 경우
        current_target = noise_rir_conv_source{5, ang}; % 5번 마이크 신호 유지 [샘플수 x 1]
    else % 그 외 모든 각도
        current_target = zeros(size(noise_rir_conv_source{5, ang})); % 0으로 만듦 [샘플수 x 1]
    end

    % 데이터를 아래로 계속 쌓음
    X_train = [X_train; current_input];
    Y_train = [Y_train; current_target];

end
% X_train=X_train';
% Y_train=Y_train';


learning = 10;
maxEpochs = learning;     %エポック数（学習回数）                      <------ 学習回数
maxEpochs2 = learning;

inputSize = num_mics;      %入力数
outputSize = 64;  %中間層のユニット数                       <-------隠れユニット数 
numResponses = 1;   %全結合層の出力層


% === Ttrainnet ===%
options = trainingOptions("adam", ... 
    InitialLearnRate = 0.01, ... % 正規化されているので高めでもOK
    MaxEpochs = maxEpochs, ...
    miniBatchSize = 2048, ...    
    GradientThreshold = 1, ...
    Metrics = "rmse", ...
    Plots="training-progress", ...
    Shuffle = 'every-epoch', ...
    ExecutionEnvironment = "auto", ...
    Verbose=0);

global_max = max(abs(X_train(:)));

X_train = X_train / global_max;
Y_train = Y_train / global_max;

%Layerの構成
layers = [ ...
    featureInputLayer(inputSize)

    fullyConnectedLayer(outputSize, Bias = zeros(outputSize, 1), BiasLearnRateFactor = 0)
    reluLayer

    fullyConnectedLayer(outputSize, Bias = zeros(outputSize, 1), BiasLearnRateFactor = 0)
    reluLayer

    fullyConnectedLayer(numResponses, Bias = 0, BiasLearnRateFactor = 0)
    ]


net = trainnet(X_train, Y_train, layers, "mse", options);
save('NNBF_learningdata_',net,global_max)