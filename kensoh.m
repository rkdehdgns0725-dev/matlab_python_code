noise_matrix_white = single(randn(sample_length, num_sources)); 
noise_matrix_pink = single(pinknoise(sample_length, num_sources));
brown = dsp.ColoredNoise('Color', 'brown', 'SamplesPerFrame', sample_length, 'NumChannels', 1);
noise_matrix_brown = single(brown());
noise_matrix_brown2=single(brown());
power=sum((noise_matrix_brown2-noise_matrix_brown).^2);
power