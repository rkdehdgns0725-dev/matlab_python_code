% Delay And Sum
% 전치 확인: source_pos_t (3 x 19), mic_pos (3 x 8)
beam_target=10;

source_pos_t = source_pos.'; 
co_micarray = mean(mic_pos, 2); % 마이크 어레이 중심 (3 x 1)

num_mics = size(mic_pos, 2);     % 8
num_sources = size(source_pos_t, 2); % 19
mic_delay = zeros(num_mics, num_sources);

for i = 1:num_sources
    % i번째 음원 위치 (3 x 1)
    current_source = source_pos_t(:, i);

    % 중심점에서 음원을 향하는 단위 벡터 (3 x 1)
    direction_vec = current_source - co_micarray;
    co_micarray_uv = direction_vec / norm(direction_vec);

    for j = 1:num_mics
        micon_vec = current_source - mic_pos(:, j);%마이크-음원방향벡터
        delay_time = dot(micon_vec, co_micarray_uv) / c;%마이크간 시간지연
        mic_delay(j, i) = delay_time;
    end
end

delay_sample=round(fs*mic_delay);


gouseionseibako=cell(size(mic_ch,1));
max_sample_delay=max(delay_sample(:,beam_target))

for i=1:mic_ch
    minimi=min(length(noise_rir_conv_source{i,1}),length(t_sound_rir_conv_source{i,1}));%%그냥 사이즈맞게 제로패딩해버리기?
    gouseionseibako{i,1}=noise_rir_conv_source{i,1}(1:minimi)+t_sound_rir_conv_source{i,1}(1:minimi);%
end
hikakuyou=(mean(cell2mat(gouseionseibako'),2));

dna_onsei=cell(num_mics,1);
for i=1:num_mics
    for j=1:num_sources
        delayed_signal = [zeros(max_sample_delay-delay_sample(i,j), 1); gouseionseibako{i,1}];
        dna_onsei{i,1} = delayed_signal(1:length(gouseionseibako{i,1}));
    end
end
saishuonsei=(mean(cell2mat(dna_onsei'),2));
max(saishuonsei)