% =========================================================================
% Script: Generate IEEE Publication Graphs (3-Threshold Edition)
% =========================================================================
disp('--- GENERATING IEEE PUBLICATION GRAPHS ---');

% Define our three operating points
t_triage = 0.30; % Custom Safety Triage
t_f1     = 0.37; % F1 Mathematical Optimum
t_diag   = 0.50; % Youden's J Diagnostic Optimum

% Extract raw scores from the trained model
[~, scores] = predict(probModel, MasterData(:, {'Entropy', 'Centroid'}));
probs = scores(:, 2);
actuals = MasterData.AnomalyLabel;

% --- FIGURE 1: BOXPLOT ---
figure('Name', 'Figure 1: Boxplot', 'Color', 'w');
boxplot(MasterData.Entropy, actuals, 'Labels', {'Healthy (0)', 'Anomaly (1)'});
title('Distribution of Spectral Entropy in Respiratory Cycles');
ylabel('Spectral Entropy (Bits)'); xlabel('Clinical Classification');
set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure1_Boxplot.png', 'Resolution', 300);

% --- FIGURE 2: SCATTER PLOT ---
figure('Name', 'Figure 2: Scatter Plot', 'Color', 'w');
gscatter(MasterData.Entropy, MasterData.Centroid, actuals, 'bg', 'os');
title('Feature Space Distribution');
xlabel('Spectral Entropy (Chaos)'); ylabel('Spectral Centroid (Pitch / Hz)');
legend('Healthy', 'Anomaly', 'Location', 'best');
grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure2_Scatter.png', 'Resolution', 300);

% --- FIGURE 3: SPECTROGRAMS ---
figure('Name', 'Figure 3: Spectrograms', 'Color', 'w', 'Position', [100, 100, 900, 400]);
subplot(1,2,1);
imagesc(10*log10(V + 1e-10)); axis xy; colormap('jet'); colorbar;
title('Raw STFT Spectrogram (Matrix V)');
xlabel('Time Frames'); ylabel('Frequency Bins');
subplot(1,2,2);
imagesc(10*log10(V_clean + 1e-10)); axis xy; colormap('jet'); colorbar;
title('Reconstructed Anomaly Layer (V_{clean})');
xlabel('Time Frames'); ylabel('Frequency Bins');
exportgraphics(gcf, 'Figure3_Spectrograms.png', 'Resolution', 300);

% --- HELPER FUNCTION TO GET METRICS FOR CURVE MARKERS ---
getMetrics = @(t) [...
    sum(probs >= t & actuals == 0) / sum(actuals == 0), ... % FPR (1 - Specificity)
    sum(probs >= t & actuals == 1) / sum(actuals == 1), ... % TPR (Recall)
    sum(probs >= t & actuals == 1) / sum(probs >= t)];      % Precision

m_triage = getMetrics(t_triage);
m_f1     = getMetrics(t_f1);
m_diag   = getMetrics(t_diag);

% --- FIGURE 4: ROC CURVE (WITH MARKERS) ---
[Xroc, Yroc, ~, AUCroc] = perfcurve(actuals, probs, 1);
figure('Name', 'Figure 4: ROC Curve', 'Color', 'w');
plot(Xroc, Yroc, 'b', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--'); % Random guess line
% Add threshold markers
plot(m_triage(1), m_triage(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(m_f1(1), m_f1(2), 'gs', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(m_diag(1), m_diag(2), 'md', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
title(['ROC Curve (AUC = ', num2str(AUCroc, '%.3f'), ')']);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
legend('ROC Curve', 'Random Guess', 'Triage (0.30)', 'F1 Opt (0.37)', 'Diag Opt (0.50)', 'Location', 'southeast');
grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure4_ROC.png', 'Resolution', 300);

% --- FIGURE 5: FEATURE IMPORTANCE ---
pValues = probModel.Coefficients.pValue(2:end);
importance = 1 - pValues; 
figure('Name', 'Figure 5: Feature Importance', 'Color', 'w');
bar(categorical({'Entropy', 'Centroid'}), importance, 'FaceColor', '#2B8CBE');
title('Feature Contribution to Diagnosis (1 - pValue)');
ylabel('Relative Statistical Importance');
ylim([0 1.05]); grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure5_Importance.png', 'Resolution', 300);

% --- FIGURE 6: TRIPLE CONFUSION MATRIX ---
figure('Name', 'Figure 6: Triple Confusion Matrix', 'Color', 'w', 'Position', [50, 100, 1400, 400]);

subplot(1, 3, 1);
cm1 = confusionchart(categorical(actuals), categorical(double(probs >= t_triage)));
cm1.Title = sprintf('Triage (0.30)'); cm1.Normalization = 'row-normalized';

subplot(1, 3, 2);
cm2 = confusionchart(categorical(actuals), categorical(double(probs >= t_f1)));
cm2.Title = sprintf('F1 Optimum (0.37)'); cm2.Normalization = 'row-normalized';

subplot(1, 3, 3);
cm3 = confusionchart(categorical(actuals), categorical(double(probs >= t_diag)));
cm3.Title = sprintf('Diagnostic (0.50)'); cm3.Normalization = 'row-normalized';

exportgraphics(gcf, 'Figure6_TripleConfusion.png', 'Resolution', 300);

% --- FIGURE 7: LEARNING CURVE ---
disp('Simulating Learning Curve Data...');
trainSizes = round(linspace(100, height(MasterData)*0.8, 10));
accCurve = zeros(length(trainSizes), 1);
for k = 1:length(trainSizes)
    idx = randperm(height(MasterData), trainSizes(k));
    miniTrain = MasterData(idx, :);
    try
        miniMdl = fitglm(miniTrain, 'AnomalyLabel ~ Entropy + Centroid', 'Distribution', 'binomial');
        miniPreds = (predict(miniMdl, MasterData) >= t_diag);
        accCurve(k) = sum(miniPreds == actuals) / height(MasterData);
    catch
        accCurve(k) = NaN;
    end
end
figure('Name', 'Figure 7: Learning Curve', 'Color', 'w');
plot(trainSizes, accCurve, '-o', 'LineWidth', 2, 'MarkerSize', 6);
title('Model Learning Curve (Stability Proof)');
xlabel('Training Sample Size'); ylabel('Prediction Accuracy (Threshold 0.50)');
grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure7_LearningCurve.png', 'Resolution', 300);

% --- FIGURE 8: PRECISION-RECALL (PR) CURVE (WITH MARKERS) ---
[Xpr, Ypr, ~, AUCpr] = perfcurve(actuals, probs, 1, 'xCrit', 'reca', 'yCrit', 'prec');
figure('Name', 'Figure 8: Precision-Recall Curve', 'Color', 'w');
plot(Xpr, Ypr, 'LineWidth', 2, 'Color', '#D95319'); hold on;
plot(m_triage(2), m_triage(3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(m_f1(2), m_f1(3), 'gs', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(m_diag(2), m_diag(3), 'md', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
title(['Precision-Recall Curve (AUC = ', num2str(AUCpr, '%.3f'), ')']);
xlabel('Recall (Sensitivity)'); ylabel('Precision (Positive Predictive Value)');
legend('PR Curve', 'Triage (0.30)', 'F1 Opt (0.37)', 'Diag Opt (0.50)', 'Location', 'northeast');
grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure8_PR_Curve.png', 'Resolution', 300);

% --- FIGURE 9: TRIPLE CLINICAL METRICS BAR CHART ---
calcMets = @(t) [...
    sum((probs >= t) == actuals) / length(actuals), ... % Accuracy
    sum(probs >= t & actuals == 1) / sum(probs >= t), ... % Precision
    sum(probs >= t & actuals == 1) / sum(actuals == 1), ... % Recall
    2 * ((sum(probs >= t & actuals == 1) / sum(probs >= t)) * (sum(probs >= t & actuals == 1) / sum(actuals == 1))) / ((sum(probs >= t & actuals == 1) / sum(probs >= t)) + (sum(probs >= t & actuals == 1) / sum(actuals == 1))) % F1
];

metricsData = [calcMets(t_triage); calcMets(t_f1); calcMets(t_diag)]';
metricNames = categorical({'Accuracy', 'Precision', 'Recall', 'F1-Score'});
metricNames = reordercats(metricNames, {'Accuracy', 'Precision', 'Recall', 'F1-Score'});

figure('Name', 'Figure 9: Triple Clinical Metrics', 'Color', 'w');
b = bar(metricNames, metricsData);
b(1).FaceColor = '#0072BD'; % Blue
b(2).FaceColor = '#77AC30'; % Green
b(3).FaceColor = '#D95319'; % Orange
title('Performance Analysis Across Operating Thresholds');
ylabel('Score (0 to 1)'); ylim([0 1.2]); 
legend('Custom Triage (0.30)', 'F1 Optimum (0.37)', 'Diagnostic (0.50)', 'Location', 'northeast');
grid on; set(gca, 'FontSize', 12);
exportgraphics(gcf, 'Figure9_MetricsBar.png', 'Resolution', 300);

disp('SUCCESS! All 9 IEEE standard graphs have been updated for 3 Thresholds.');
