close all;

acc_woNoise = 97.0;

noiseAmpRatioVec = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.5];

acc_wNoise_rand = [97, 97, 97, 97, 97, 97, 96.8, 96.4, 91.4, 78.8, 67.8, 43.6, 34.6];
acc_wNoise_FGM = [96.8, 95, 90.5, 83.4, 75.3, 53.7, 32.8, 17.4, 9.1, 8.6, 8.2, 8.1, 7.8];

figure;
yline(acc_woNoise, '--', 'LineWidth', 3); hold on;
semilogx(10*log10(noiseAmpRatioVec), acc_wNoise_rand, '--v', 'MarkerSize', 20, 'LineWidth', 3);
semilogx(10*log10(noiseAmpRatioVec), acc_wNoise_FGM, '--x', 'MarkerSize', 20, 'LineWidth', 3);
ylim([0 100]);
grid on;

xlabel('Amplitude ratio of signal and noise (dB)');
ylabel('Accuracy (%)');

legend({'Noise-free', ...
    'Random noise', ...
    'FGM noise'});
