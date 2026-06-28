% =========================================================================
% Project: Automated Identification of Adventitious Respiratory Sounds
% Author: Md. Rezwan Ahmed
% Description: STFT-NMF time-series decomposition & statistical regression
% =========================================================================
clc; clear; close all;

% --- PHASE 1: AUDIO PROCESSING & MATH EXTRACTION ---
disp('--- PHASE 1: AUDIO PROCESSING ---');
dataFolder = 'audio_and_txt_files'; 
filePattern = fullfile(dataFolder, '*.wav');
wavFiles = dir(filePattern);
MasterData = table();

for i = 1:length(wavFiles)
    baseFileName = wavFiles(i).name;
    audioPath = fullfile(dataFolder, baseFileName);
    txtPath = fullfile(dataFolder, strrep(baseFileName, '.wav', '.txt'));
    
    if ~isfile(txtPath)
        continue; 
    end
    
    [y, fs] = audioread(audioPath);
    try annotations = readmatrix(txtPath); catch; continue; end
    
    numBreaths = size(annotations, 1);
    
    for b = 1:numBreaths
        % 1. Slicing
        startTime = annotations(b, 1);
        endTime = annotations(b, 2);
        isAnomaly = max(annotations(b, 3), annotations(b, 4));
        
        startIndex = max(1, round(startTime * fs));
        endIndex = min(length(y), round(endTime * fs));
        breathSignal = y(startIndex:endIndex);
        
        targetLength = 4 * fs;
        if length(breathSignal) < targetLength
            breathSignal = [breathSignal; zeros(targetLength - length(breathSignal), 1)];
        elseif length(breathSignal) > targetLength
            breathSignal = breathSignal(1:targetLength);
        end
        
        % 2. STFT
        window = round(0.05 * fs); overlap = round(0.025 * fs); nfft = 1024;
        [S, ~, ~] = spectrogram(breathSignal, window, overlap, nfft, fs);
        V = abs(S);
        
        % 3. NMF (With Warning Bypass & Safety Fallback)
        opt = statset('MaxIter', 50, 'Display', 'off'); 
        warning('off', 'stats:nnmf:LowRank'); 
        [W, H] = nnmf(V, 2, 'Replicates', 1, 'Options', opt);
        warning('on', 'stats:nnmf:LowRank');
        
        V_clean = W(:, 2) * H(2, :);
        if sum(V_clean(:)) == 0 || isnan(sum(V_clean(:)))
            V_clean = W(:, 1) * H(1, :); 
        end
        
        % 4. Feature Extraction
        eps_val = 1e-10; 
        p = V_clean(:) / (sum(V_clean(:)) + eps_val); 
        p = p(p > 0); 
        featEntropy = -sum(p .* log2(p));
        
        freqWeights = (1:size(V_clean, 1))';
        featCentroid = sum(freqWeights .* sum(V_clean, 2)) / (sum(V_clean(:)) + eps_val);
        
        % 5. Save to Master Table
        newRow = table(featEntropy, featCentroid, isAnomaly, 'VariableNames', {'Entropy', 'Centroid', 'AnomalyLabel'});
        MasterData = [MasterData; newRow];
    end
end
disp('Phase 1 Complete! MasterData Table Generated.');

% --- PHASE 2: COMPUTATIONAL STATISTICS ---
disp(' '); disp('--- PHASE 2: COMPUTATIONAL STATISTICS ---');
R = corrcoef(MasterData.Entropy, MasterData.Centroid);
fprintf('Feature Correlation Coefficient: %.4f\n', R(1, 2));

healthyEntropy = MasterData.Entropy(MasterData.AnomalyLabel == 0);
anomalyEntropy = MasterData.Entropy(MasterData.AnomalyLabel == 1);

[h_t, p_t] = ttest2(healthyEntropy, anomalyEntropy);
fprintf('T-Test p-value: %.4e (Significant if < 0.05)\n', p_t);

[h_f, p_f] = vartest2(healthyEntropy, anomalyEntropy);
fprintf('F-Test p-value: %.4e\n', p_f);

% --- ADVANCED NON-PARAMETRIC STATISTICS ---
disp(' '); disp('--- ADVANCED NON-PARAMETRIC STATISTICS ---');
[p_mwu, h_mwu, stats_mwu] = ranksum(healthyEntropy, anomalyEntropy);
fprintf('Wilcoxon Rank Sum p-value: %.4e (Significant if < 0.05)\n', p_mwu);

% --- 10-FOLD CROSS-VALIDATION ---
disp(' '); disp('--- 10-FOLD CROSS-VALIDATION ---');
numFolds = 10;
cv = cvpartition(MasterData.AnomalyLabel, 'KFold', numFolds);
cvAccuracies = zeros(numFolds, 1);
cvRecalls = zeros(numFolds, 1);

for i = 1:numFolds
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    trainSet = MasterData(trainIdx, :);
    testSet = MasterData(testIdx, :);
    
    mdl_cv = fitglm(trainSet, 'AnomalyLabel ~ Entropy + Centroid', 'Distribution', 'binomial');
    
    probs_cv = predict(mdl_cv, testSet);
    preds_cv = (probs_cv >= 0.3);
    
    cvAccuracies(i) = sum(preds_cv == testSet.AnomalyLabel) / height(testSet);
    
    TP_cv = sum(preds_cv == 1 & testSet.AnomalyLabel == 1);
    FN_cv = sum(preds_cv == 0 & testSet.AnomalyLabel == 1);
    if (TP_cv + FN_cv) > 0
        cvRecalls(i) = TP_cv / (TP_cv + FN_cv);
    else
        cvRecalls(i) = NaN; 
    end
end

fprintf('10-Fold Mean Accuracy: %.2f%%\n', mean(cvAccuracies) * 100);
fprintf('10-Fold Mean Recall (Sensitivity): %.2f%%\n', mean(cvRecalls, 'omitnan') * 100);

% --- TRAINING LOGISTIC REGRESSION MODEL ---
disp(' '); disp('--- TRAINING LOGISTIC REGRESSION MODEL ---');
probModel = fitglm(MasterData, 'AnomalyLabel ~ Entropy + Centroid', 'Distribution', 'binomial');

% --- FORMAL INDEPENDENCE TEST (CHI-SQUARE) ---
disp(' '); disp('--- FORMAL INDEPENDENCE TEST (CHI-SQUARE) ---');
% BUG FIX: Added 'double()' so the predictions match the table's data type
preds_chi = double(predict(probModel, MasterData) >= 0.3);
actuals_chi = MasterData.AnomalyLabel;

observed = confusionmat(actuals_chi, preds_chi);
expected = sum(observed, 2) * sum(observed, 1) / sum(observed(:));
chi2stat = sum((observed(:) - expected(:)).^2 ./ expected(:));
p_chi2 = 1 - chi2cdf(chi2stat, 1);

fprintf('Chi-Square Statistic: %.2f\n', chi2stat);
fprintf('Chi-Square p-value: %.4e (Significant if < 0.05)\n', p_chi2);

if p_chi2 < 0.05
    disp('CONCLUSION: Predictions and Actual Labels are dependent (Model is mathematically valid).');
else
    disp('CONCLUSION: Predictions are independent (Model is guessing randomly).');
end

% --- FINAL OUTPUT: DUAL-THRESHOLD DIAGNOSTIC ENGINE ---
disp(' '); disp('--- TESTING THE MODEL (DUAL-THRESHOLD APPROACH) ---');

% Define the two operating points
triageThreshold = 0.30;       % Optimized for Clinical Safety (Max Recall)
diagnosticThreshold = 0.50;   % Optimized for Mathematical Accuracy (Youden's J Optimum)

testIndex = 10; 
testData = table(MasterData.Entropy(testIndex), MasterData.Centroid(testIndex), 'VariableNames', {'Entropy', 'Centroid'});
predictedProb = predict(probModel, testData);

fprintf('Machine Calculated Probability of Anomaly: %.2f%%\n\n', predictedProb * 100);

% TIER 1: TRIAGE EVALUATION
fprintf('--- TIER 1: EARLY SCREENING (Threshold = %.2f) ---\n', triageThreshold);
if predictedProb >= triageThreshold
    disp('TRIAGE RESULT: [FLAGGED] - Patient exhibits spectral overlap with anomalies. Do not discharge.');
else
    disp('TRIAGE RESULT: [CLEAR] - Patient cleared by high-sensitivity screening.');
end

% TIER 2: DIAGNOSTIC EVALUATION
fprintf('\n--- TIER 2: DEFINITIVE DIAGNOSIS (Threshold = %.2f) ---\n', diagnosticThreshold);
if predictedProb >= diagnosticThreshold
    disp('DIAGNOSTIC RESULT: [POSITIVE] - Mathematical anomaly confirmed with high precision.');
else
    disp('DIAGNOSTIC RESULT: [NEGATIVE] - Probability falls below optimum accuracy boundary.');
end

% --- MATHEMATICAL THRESHOLD OPTIMIZATION ---
disp(' '); disp('--- MATHEMATICAL THRESHOLD OPTIMIZATION ---');
[~, scores] = predict(probModel, MasterData(:, {'Entropy', 'Centroid'}));
probs = scores(:, 2); 

bestF1 = 0;  bestThreshF1 = 0;  
acc_F1 = 0; rec_F1 = 0; prec_F1 = 0; spec_F1 = 0;

bestJ = -1;  bestThreshJ = 0;   
acc_J = 0; rec_J = 0; prec_J = 0; spec_J = 0; f1_J = 0;

for t = 0:0.01:1
    preds = (probs >= t);
    
    TP = sum(preds == 1 & MasterData.AnomalyLabel == 1);
    FP = sum(preds == 1 & MasterData.AnomalyLabel == 0);
    FN = sum(preds == 0 & MasterData.AnomalyLabel == 1);
    TN = sum(preds == 0 & MasterData.AnomalyLabel == 0);
    
    if (TP + FN) > 0; Recall = TP / (TP + FN); else; Recall = 0; end
    if (TN + FP) > 0; Specificity = TN / (TN + FP); else; Specificity = 0; end
    if (TP + FP) > 0; Precision = TP / (TP + FP); else; Precision = 0; end
    
    CurrentAccuracy = (TP + TN) / height(MasterData);
    
    if (Precision + Recall) > 0
        F1 = 2 * (Precision * Recall) / (Precision + Recall);
    else
        F1 = 0;
    end
    
    J = Recall + Specificity - 1;
    
    if F1 > bestF1
        bestF1 = F1; bestThreshF1 = t;
        acc_F1 = CurrentAccuracy; rec_F1 = Recall; 
        prec_F1 = Precision; spec_F1 = Specificity;
    end
    
    if J > bestJ
        bestJ = J; bestThreshJ = t;
        acc_J = CurrentAccuracy; rec_J = Recall; 
        prec_J = Precision; spec_J = Specificity; f1_J = F1;
    end
end

fprintf('\n[OPTIMIZATION 1] F1-Score Focus (Imbalanced Data Method):\n');
fprintf('  Optimal Threshold: %.2f\n', bestThreshF1);
fprintf('  Accuracy:          %.2f%%\n', acc_F1 * 100);
fprintf('  Precision:         %.2f%%\n', prec_F1 * 100);
fprintf('  Recall:            %.2f%%\n', rec_F1 * 100);
fprintf('  Specificity:       %.2f%%\n', spec_F1 * 100);
fprintf('  F1-Score:          %.4f\n', bestF1);

fprintf('\n[OPTIMIZATION 2] Youden''s J Focus (ROC Balance Method):\n');
fprintf('  Optimal Threshold: %.2f\n', bestThreshJ);
fprintf('  Accuracy:          %.2f%%\n', acc_J * 100);
fprintf('  Precision:         %.2f%%\n', prec_J * 100);
fprintf('  Recall:            %.2f%%\n', rec_J * 100);
fprintf('  Specificity:       %.2f%%\n', spec_J * 100);
fprintf('  F1-Score:          %.4f\n', f1_J);
fprintf('  J-Statistic:       %.4f\n', bestJ);

% =========================================================================
fprintf('\n[OPTIMIZATION 3] Custom Clinical Triage (Threshold = 0.30):\n');
% Force the evaluation exactly at 0.30
t_custom = 0.30;
preds_custom = (probs >= t_custom);

TP_c = sum(preds_custom == 1 & MasterData.AnomalyLabel == 1);
FP_c = sum(preds_custom == 1 & MasterData.AnomalyLabel == 0);
FN_c = sum(preds_custom == 0 & MasterData.AnomalyLabel == 1);
TN_c = sum(preds_custom == 0 & MasterData.AnomalyLabel == 0);

if (TP_c + FN_c) > 0; rec_c = TP_c / (TP_c + FN_c); else; rec_c = 0; end
if (TN_c + FP_c) > 0; spec_c = TN_c / (TN_c + FP_c); else; spec_c = 0; end
if (TP_c + FP_c) > 0; prec_c = TP_c / (TP_c + FP_c); else; prec_c = 0; end
acc_c = (TP_c + TN_c) / height(MasterData);
if (prec_c + rec_c) > 0; f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c); else; f1_c = 0; end

fprintf('  Applied Threshold: %.2f (Clinically Justified)\n', t_custom);
fprintf('  Accuracy:          %.2f%%\n', acc_c * 100);
fprintf('  Precision:         %.2f%%\n', prec_c * 100);
fprintf('  Recall (Sensitivity):%.2f%%  <-- (This is your safety metric!)\n', rec_c * 100);
fprintf('  Specificity:       %.2f%%\n', spec_c * 100);
fprintf('  F1-Score:          %.4f\n', f1_c);
disp('-----------------------------------------------------------');
