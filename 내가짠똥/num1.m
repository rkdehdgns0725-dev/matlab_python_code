ini='C:\Users\hsm15\OneDrive - 창원대학교\デスクトップ\水町研究室\R8_修士論文研究データ_243D7023_中島貫\研究で使用する音源ファイル\直線_8ch_等間隔1cm2025_06_12\0'
Fs=96000;%Hz
ts=2.5;
audio_sample_num=Fs*ts;

audi_data = zeros(audio_sample_num, 8); % 배열 사전 할당


for i = 1:8
    fileName = sprintf('%s\\ch%d.wav', ini, i);
    matr=audioread(fileName);
    audi_data(:, i) = matr(1:audio_sample_num, 1); % Store the audio data for channel i
end

    
