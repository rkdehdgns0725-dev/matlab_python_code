% figure();
% plot(X_train_final(250:1000, 5), 'b', 'LineWidth', 1.5)  % 파란색 선으로 입력 채널 5 플롯
% hold on
% plot(Y_train_final(250:1000, 1), 'r--', 'LineWidth', 1.5)    % 빨간색 점선으로 정답 T 플롯
% hold off
% legend('Input Ch5', 'Target T')
% sum((X_train_final(:, 5)-Y_train_final(:, 1)).^2,"all")



% % Band 1 (0~500Hz)의 1번 각도와 10번 각도 파형 비교
% sig_angle_1 = bandpassed_dataset{5,1}{1,1};
% sig_angle_10 = bandpassed_dataset{5,10}{1,1};
% 
% figure;
% plot(sig_angle_1(1:1000)); % 앞부분 일부 샘플만 확대
% hold on;
% plot(sig_angle_10(1:1000), '--');
% legend('-90도 (1번)', '0도 (10번)');
% title('Band 1 (0~500Hz) 파형 비교');
% sum((bandpassed_dataset{5,1}{1,1}-bandpassed_dataset{5,10}{1,1}).^2,"all")

