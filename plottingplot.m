figure();
plot(X_train_final(250:1000, 5), 'b', 'LineWidth', 1.5)  % 파란색 선으로 입력 채널 5 플롯
hold on
plot(Y_train_final(250:1000, 1), 'r--', 'LineWidth', 1.5)    % 빨간색 점선으로 정답 T 플롯
hold off
legend('Input Ch5', 'Target T')
sum((X_train_final(:, 5)-Y_train_final(:, 1)).^2,"all")