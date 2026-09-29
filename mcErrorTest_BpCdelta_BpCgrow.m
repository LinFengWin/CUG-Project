function results = mcErrorTest_BpCdelta_BpCgrow(POP, Builtup, varargin)
%% ============================================================
%  mcErrorTest_BpCdelta_BpCgrow
%  Monte Carlo error test for BpCdelta and BpCgrow
%
%  Usage:
%    results = mcErrorTest_BpCdelta_BpCgrow(POP, Builtup)
%    results = mcErrorTest_BpCdelta_BpCgrow(POP, Builtup, 'epsBuiltup', 0.10)
%    results = mcErrorTest_BpCdelta_BpCgrow(POP, Builtup, ...
%                'epsBuiltup', 0.10, 'epsPOP', 0.03, 'N', 10000)
%
%  Inputs:
%    POP      - n x T population matrix (rows = cities, cols = time points)
%    Builtup  - n x T built-up land matrix (same size as POP)
%
%  Optional name-value pairs:
%    'epsBuiltup' - relative error of built-up land, default 0.10
%    'epsPOP'     - relative error of population,  default 0.03
%    'N'          - number of Monte Carlo simulations, default 10000
%    'exportDir'  - output directory, default 'mc_error_results'
%    'export'     - whether to export results, default true
%    'plot'       - whether to plot, default true
%
%  Output:
%    results - struct containing:
%       .BpCdelta_true    - true BpCdelta matrix (n x T-1)
%       .BpCgrow_true     - true BpCgrow matrix (n x T-1)
%       .BpCdelta_sim     - BpCdelta MC results (N x n x T-1)
%       .BpCgrow_sim      - BpCgrow MC results (N x n x T-1)
%       .BpCdelta_stats   - BpCdelta error statistics
%       .BpCgrow_stats    - BpCgrow error statistics
%       .BpCdelta_ci      - BpCdelta 95% confidence interval
%       .BpCgrow_ci       - BpCgrow 95% confidence interval
%       .BpCdelta_normality - BpCdelta normality test results
%       .BpCgrow_normality  - BpCgrow normality test results
%       .BpCdelta_pdf     - BpCdelta kernel density estimate
%       .BpCgrow_pdf      - BpCgrow kernel density estimate
%
%  Note:
%    epsBuiltup and epsPOP are INPUT relative errors.
%    BpCdelta_stats.rel_std and BpCgrow_stats.rel_std are OUTPUT relative errors.
%    They are NOT the same quantity.
%
%    Samples whose original Builtup values are less than 0.1, or whose
%    original POP values are less than 1000, are excluded from all
%    computations.
% ============================================================

%% 1. Parse input arguments
p = inputParser;
addRequired(p, 'POP', @(x) isnumeric(x) && ismatrix(x));
addRequired(p, 'Builtup', @(x) isnumeric(x) && ismatrix(x));
addParameter(p, 'epsBuiltup', 0.10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'epsPOP', 0.03, @(x) isnumeric(x) && x > 0);
addParameter(p, 'N', 100, @(x) isnumeric(x) && x > 0 && mod(x,1)==0);
addParameter(p, 'exportDir', 'mc_error_results', @ischar);
addParameter(p, 'export', true, @islogical);
addParameter(p, 'plot', true, @islogical);

parse(p, POP, Builtup, varargin{:});

eps_B     = p.Results.epsBuiltup;
eps_P     = p.Results.epsPOP;
N         = p.Results.N;
exportDir = p.Results.exportDir;
doExport  = p.Results.export;
doPlot    = p.Results.plot;

%% 2. Data validation
if ~isequal(size(POP), size(Builtup))
    error('POP and Builtup must have the same size.');
end

[n, T] = size(POP);

if T < 2
    error('The time series must contain at least 2 time points.');
end

% 检查非正值
if any(POP(:) <= 0) || any(Builtup(:) <= 0)
    warning('POP or Builtup contains non-positive values, replaced by 1e-6.');
    POP = max(POP, 1e-6);
    Builtup = max(Builtup, 1e-6);
end

%% 2b. 排除原始数据不满足阈值的样本
% Builtup 原始值 < 0.1 不参与计算
% POP     原始值 < 1000 不参与计算
% 对每个城市 i 和每个时间步对 (t, t+1)，只有当
% Builtup(i,t)、Builtup(i,t+1) 都 >= 0.1，
% 且 POP(i,t)、POP(i,t+1) 都 >= 1000 时，该样本才参与计算。
minBuiltup = 0.1;
minPOP     = 1000;

valid_pair = false(n, T-1);
for i = 1:n
    for t = 1:T-1
        if Builtup(i, t)   >= minBuiltup && Builtup(i, t+1) >= minBuiltup && ...
           POP(i, t)       >= minPOP     && POP(i, t+1)     >= minPOP
            valid_pair(i, t) = true;
        end
    end
end

fprintf('Valid sample pairs (Builtup >= %.2f and POP >= %.0f): %d / %d (%.2f%%)\n', ...
    minBuiltup, minPOP, sum(valid_pair(:)), n*(T-1), ...
    100 * sum(valid_pair(:)) / (n*(T-1)));

%% 3. Compute true indicator values
% BpC_t = Builtup / POP * 1e6
BpC_true = Builtup ./ POP * 1000000;    % n x T

% BpCdelta = BpC_t2 - BpC_t1
BpCdelta_true = nan(n, T-1);            % n x (T-1)
for i = 1:n
    for t = 1:T-1
        if valid_pair(i, t)
            BpCdelta_true(i, t) = BpC_true(i, t+1) - BpC_true(i, t);
        end
    end
end

% BpCgrow = (Urb_t2 / Urb_t1) / (POP_t2 / POP_t1)
BpCgrow_true = nan(n, T-1);             % n x (T-1)
for i = 1:n
    for t = 1:T-1
        if valid_pair(i, t)
            num = Builtup(i, t+1) / Builtup(i, t);
            den = POP(i, t+1) / POP(i, t);
            if abs(den) > 1e-6
                BpCgrow_true(i, t) = num / den;
            end
        end
    end
end

%% 4. Monte Carlo simulation
fprintf('Starting Monte Carlo simulation, N = %d ...\n', N);

BpCdelta_sim = nan(N, n, T-1);
BpCgrow_sim  = nan(N, n, T-1);

nan_BpCgrow_count = 0;

for k = 1:N
    % Add relative errors to each input variable
    % 使用截断正态分布，保证正值
    Builtup_s = Builtup .* max(1 + eps_B * randn(n, T), 1e-6);
    POP_s     = POP     .* max(1 + eps_P * randn(n, T), 1e-6);

    % BpC_t
    BpC_sim = Builtup_s ./ POP_s * 1000000;   % n x T

    % BpCdelta and BpCgrow
    for i = 1:n
        for t = 1:T-1
            if ~valid_pair(i, t)
                continue;   % 原始数据不满足阈值，跳过
            end

            % BpCdelta = BpC_t2 - BpC_t1
            BpCdelta_sim(k, i, t) = BpC_sim(i, t+1) - BpC_sim(i, t);

            % BpCgrow = (Urb_t2 / Urb_t1) / (POP_t2 / POP_t1)
            num = Builtup_s(i, t+1) / Builtup_s(i, t);
            den = POP_s(i, t+1) / POP_s(i, t);
            if abs(den) > 1e-6
                BpCgrow_sim(k, i, t) = num / den;
            else
                nan_BpCgrow_count = nan_BpCgrow_count + 1;
            end
        end
    end

    if mod(k, 1000) == 0
        fprintf('  Completed %d / %d\n', k, N);
    end
end

fprintf('Monte Carlo simulation finished.\n');
fprintf('BpCgrow NaN count = %d (%.4f%%)\n', ...
    nan_BpCgrow_count, 100 * nan_BpCgrow_count / (N * n * (T-1)));

%% 5. Error computation (using repmat to expand true values)
% BpCdelta error
BpCdelta_true_3d = repmat(reshape(BpCdelta_true, [1, n, T-1]), [N, 1, 1]);
BpCdelta_err     = BpCdelta_sim - BpCdelta_true_3d;
BpCdelta_err_vec = BpCdelta_err(:);
BpCdelta_err_vec = BpCdelta_err_vec(~isnan(BpCdelta_err_vec));

% BpCgrow error
BpCgrow_true_3d = repmat(reshape(BpCgrow_true, [1, n, T-1]), [N, 1, 1]);
BpCgrow_err     = BpCgrow_sim - BpCgrow_true_3d;
BpCgrow_err_vec = BpCgrow_err(:);
BpCgrow_err_vec = BpCgrow_err_vec(~isnan(BpCgrow_err_vec));

% Relative errors
% BpCdelta 相对误差，剔除真值接近 0 的样本
valid_BpCdelta = abs(BpCdelta_true_3d) > 1e-6;
BpCdelta_rel_err = nan(size(BpCdelta_err));
BpCdelta_rel_err(valid_BpCdelta) = BpCdelta_err(valid_BpCdelta) ./ BpCdelta_true_3d(valid_BpCdelta);
BpCdelta_rel_err_vec = BpCdelta_rel_err(:);
BpCdelta_rel_err_vec = BpCdelta_rel_err_vec(~isnan(BpCdelta_rel_err_vec));

% BpCgrow 相对误差，剔除真值接近 0 的样本
valid_BpCgrow = abs(BpCgrow_true_3d) > 1e-6;
BpCgrow_rel_err = nan(size(BpCgrow_err));
BpCgrow_rel_err(valid_BpCgrow) = BpCgrow_err(valid_BpCgrow) ./ BpCgrow_true_3d(valid_BpCgrow);
BpCgrow_rel_err_vec = BpCgrow_rel_err(:);
BpCgrow_rel_err_vec = BpCgrow_rel_err_vec(~isnan(BpCgrow_rel_err_vec));

%% 6. Statistics
BpCdelta_stats = computeStats(BpCdelta_err_vec, BpCdelta_rel_err_vec);
BpCgrow_stats  = computeStats(BpCgrow_err_vec,  BpCgrow_rel_err_vec);

%% 7. Confidence intervals
BpCdelta_ci = quantile(BpCdelta_err_vec, [0.025, 0.5, 0.975]);
BpCgrow_ci  = quantile(BpCgrow_err_vec,  [0.025, 0.5, 0.975]);

BpCdelta_rel_ci = quantile(BpCdelta_rel_err_vec, [0.025, 0.5, 0.975]);
BpCgrow_rel_ci  = quantile(BpCgrow_rel_err_vec,  [0.025, 0.5, 0.975]);

%% 8. Normality tests
BpCdelta_normality = normalityTest(BpCdelta_err_vec);
BpCgrow_normality  = normalityTest(BpCgrow_err_vec);

%% 9. Kernel density estimation
[BpCdelta_pdf_f, BpCdelta_pdf_x] = ksdensity(BpCdelta_err_vec);
BpCdelta_pdf.f = BpCdelta_pdf_f;
BpCdelta_pdf.x = BpCdelta_pdf_x;

[BpCgrow_pdf_f, BpCgrow_pdf_x] = ksdensity(BpCgrow_err_vec);
BpCgrow_pdf.f = BpCgrow_pdf_f;
BpCgrow_pdf.x = BpCgrow_pdf_x;

%% 10. Package results
results = struct();
results.BpCdelta_true        = BpCdelta_true;
results.BpCgrow_true         = BpCgrow_true;
results.BpCdelta_sim         = BpCdelta_sim;
results.BpCgrow_sim          = BpCgrow_sim;
results.BpCdelta_err_vec     = BpCdelta_err_vec;
results.BpCgrow_err_vec      = BpCgrow_err_vec;
results.BpCdelta_rel_err_vec = BpCdelta_rel_err_vec;
results.BpCgrow_rel_err_vec  = BpCgrow_rel_err_vec;
results.BpCdelta_stats       = BpCdelta_stats;
results.BpCgrow_stats        = BpCgrow_stats;
results.BpCdelta_ci          = BpCdelta_ci;
results.BpCgrow_ci           = BpCgrow_ci;
results.BpCdelta_rel_ci      = BpCdelta_rel_ci;
results.BpCgrow_rel_ci       = BpCgrow_rel_ci;
results.BpCdelta_normality   = BpCdelta_normality;
results.BpCgrow_normality    = BpCgrow_normality;
results.BpCdelta_pdf         = BpCdelta_pdf;
results.BpCgrow_pdf          = BpCgrow_pdf;
results.epsBuiltup           = eps_B;
results.epsPOP               = eps_P;
results.N                    = N;
results.nan_BpCgrow_count    = nan_BpCgrow_count;
results.valid_pair           = valid_pair;

%% 11. Export results
if doExport
    if ~exist(exportDir, 'dir')
        mkdir(exportDir);
    end

    fid = fopen(fullfile(exportDir, 'mc_error_summary.txt'), 'w', 'n', 'UTF-8');
    fprintf(fid, '========================================\n');
    fprintf(fid, '  Monte Carlo Error Test for BpCdelta and BpCgrow\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'Number of simulations N = %d\n', N);
    fprintf(fid, 'Relative error of built-up land epsBuiltup = %.4f\n', eps_B);
    fprintf(fid, 'Relative error of population      epsPOP     = %.4f\n', eps_P);
    fprintf(fid, 'Valid sample pairs (Builtup >= %.2f and POP >= %.0f) = %d / %d (%.2f%%)\n\n', ...
        minBuiltup, minPOP, sum(valid_pair(:)), n*(T-1), ...
        100 * sum(valid_pair(:)) / (n*(T-1)));

    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '  BpCdelta error statistics\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean              = %.6f\n', BpCdelta_stats.mean);
    fprintf(fid, 'Std               = %.6f\n', BpCdelta_stats.std);
    fprintf(fid, 'Median            = %.6f\n', BpCdelta_stats.median);
    fprintf(fid, 'MAD               = %.6f\n', BpCdelta_stats.mad);
    fprintf(fid, 'Skewness          = %.6f\n', BpCdelta_stats.skewness);
    fprintf(fid, 'Kurtosis          = %.6f\n', BpCdelta_stats.kurtosis);
    fprintf(fid, 'Rel. mean         = %.6f\n', BpCdelta_stats.rel_mean);
    fprintf(fid, 'Rel. std          = %.6f\n', BpCdelta_stats.rel_std);
    fprintf(fid, '95%% CI            = [%.6f, %.6f]\n', BpCdelta_ci(1), BpCdelta_ci(3));
    fprintf(fid, 'Relative 95%% CI   = [%.6f, %.6f]\n', BpCdelta_rel_ci(1), BpCdelta_rel_ci(3));
    fprintf(fid, 'Lilliefors p      = %.6f\n', BpCdelta_normality.lillie_p);
    fprintf(fid, 'Jarque-Bera p     = %.6f\n', BpCdelta_normality.jb_p);
    if BpCdelta_normality.lillie_p < 0.05
        fprintf(fid, 'Conclusion: BpCdelta errors are NOT normally distributed\n');
    else
        fprintf(fid, 'Conclusion: BpCdelta errors are normally distributed\n');
    end

    fprintf(fid, '\n----------------------------------------\n');
    fprintf(fid, '  BpCgrow error statistics\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean              = %.6f\n', BpCgrow_stats.mean);
    fprintf(fid, 'Std               = %.6f\n', BpCgrow_stats.std);
    fprintf(fid, 'Median            = %.6f\n', BpCgrow_stats.median);
    fprintf(fid, 'MAD               = %.6f\n', BpCgrow_stats.mad);
    fprintf(fid, 'Skewness          = %.6f\n', BpCgrow_stats.skewness);
    fprintf(fid, 'Kurtosis          = %.6f\n', BpCgrow_stats.kurtosis);
    fprintf(fid, 'Rel. mean         = %.6f\n', BpCgrow_stats.rel_mean);
    fprintf(fid, 'Rel. std          = %.6f\n', BpCgrow_stats.rel_std);
    fprintf(fid, '95%% CI            = [%.6f, %.6f]\n', BpCgrow_ci(1), BpCgrow_ci(3));
    fprintf(fid, 'Relative 95%% CI   = [%.6f, %.6f]\n', BpCgrow_rel_ci(1), BpCgrow_rel_ci(3));
    fprintf(fid, 'Lilliefors p      = %.6f\n', BpCgrow_normality.lillie_p);
    fprintf(fid, 'Jarque-Bera p     = %.6f\n', BpCgrow_normality.jb_p);
    if BpCgrow_normality.lillie_p < 0.05
        fprintf(fid, 'Conclusion: BpCgrow errors are NOT normally distributed\n');
    else
        fprintf(fid, 'Conclusion: BpCgrow errors are normally distributed\n');
    end

    fclose(fid);

    save(fullfile(exportDir, 'mc_error_results.mat'), 'results', '-v7.3');

    fprintf('Results exported to: %s\n', exportDir);
end

%% 12. Visualization
if doPlot
    figure('Position', [100, 100, 1400, 900], 'Color', 'w');

    subplot(2,3,1);
    BpCdelta_err_vec_plot = BpCdelta_err_vec(BpCdelta_err_vec >= -100 & BpCdelta_err_vec <= 100);
    histogram(BpCdelta_err_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
    plot(BpCdelta_pdf.x, BpCdelta_pdf.f, 'r-', 'LineWidth', 2);
    xlabel('?BpC absolute error');
    ylabel('Probability density');
    title('?BpC error distribution');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-100, 100]);

    subplot(2,3,2);
    BpCgrow_err_vec_plot = BpCgrow_err_vec(BpCgrow_err_vec >= -1 & BpCgrow_err_vec <= 1);
    histogram(BpCgrow_err_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
    plot(BpCgrow_pdf.x, BpCgrow_pdf.f, 'r-', 'LineWidth', 2);
    xlabel('BpCgrow absolute error');
    ylabel('Probability density');
    title('BpCgrow error distribution');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-1, 1]);

    subplot(2,3,3);
    BpCdelta_rel_err_vec_plot = BpCdelta_rel_err_vec(BpCdelta_rel_err_vec >= -5 & BpCdelta_rel_err_vec <= 5);
    histogram(BpCdelta_rel_err_vec_plot, 50, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    xlabel('BpCdelta relative error');
    ylabel('Probability density');
    title('BpCdelta relative error distribution');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-5, 5]);

    subplot(2,3,4);
    BpCgrow_rel_err_vec_plot = BpCgrow_rel_err_vec(BpCgrow_rel_err_vec >= -1 & BpCgrow_rel_err_vec <= 1);
    histogram(BpCgrow_rel_err_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    xlabel('BpCgrow relative error');
    ylabel('Probability density');
    title('BpCgrow relative error distribution');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-1, 1]);

    if doExport
        saveas(gcf, fullfile(exportDir, 'mc_error_distributions.png'));
    end
end

%% 13. Command-line output
fprintf('\n========================================\n');
fprintf('  BpCdelta Monte Carlo error test results\n');
fprintf('========================================\n');
fprintf('Mean = %.6f, Std = %.6f\n', BpCdelta_stats.mean, BpCdelta_stats.std);
fprintf('Relative mean = %.6f, Std = %.6f\n', BpCdelta_stats.rel_mean, BpCdelta_stats.rel_std);
fprintf('95%% confidence interval = [%.6f, %.6f]\n', BpCdelta_ci(1), BpCdelta_ci(3));
fprintf('Normality test p = %.6f\n', BpCdelta_normality.lillie_p);
if BpCdelta_normality.lillie_p < 0.05
    fprintf('Conclusion: BpCdelta errors are NOT normally distributed\n');
else
    fprintf('Conclusion: BpCdelta errors are normally distributed\n');
end

fprintf('\n========================================\n');
fprintf('  BpCgrow Monte Carlo error test results\n');
fprintf('========================================\n');
fprintf('Mean = %.6f, Std = %.6f\n', BpCgrow_stats.mean, BpCgrow_stats.std);
fprintf('Relative mean = %.6f, Std = %.6f\n', BpCgrow_stats.rel_mean, BpCgrow_stats.rel_std);
fprintf('95%% confidence interval = [%.6f, %.6f]\n', BpCgrow_ci(1), BpCgrow_ci(3));
fprintf('Normality test p = %.6f\n', BpCgrow_normality.lillie_p);
if BpCgrow_normality.lillie_p < 0.05
    fprintf('Conclusion: BpCgrow errors are NOT normally distributed\n');
else
    fprintf('Conclusion: BpCgrow errors are normally distributed\n');
end

end

%% ============================================================
%  Helper functions
% ============================================================

function stats = computeStats(err_vec, rel_err_vec)
    stats.mean     = mean(err_vec);
    stats.std      = std(err_vec);
    stats.median   = median(err_vec);
    stats.mad      = mad(err_vec);
    stats.skewness = skewness(err_vec);
    stats.kurtosis = kurtosis(err_vec);
    stats.min      = min(err_vec);
    stats.max      = max(err_vec);
    stats.q25      = quantile(err_vec, 0.25);
    stats.q75      = quantile(err_vec, 0.75);

    stats.rel_mean = mean(rel_err_vec);
    stats.rel_std  = std(rel_err_vec);
    stats.rel_median = median(rel_err_vec);
end

function result = normalityTest(err_vec)
    result = struct();

    [h1, p1] = lillietest(err_vec);
    result.lillie_h = h1;
    result.lillie_p = p1;

    [h2, p2] = jbtest(err_vec);
    result.jb_h = h2;
    result.jb_p = p2;

    pd = makedist('Normal', mean(err_vec), std(err_vec));
    [h3, p3] = kstest(err_vec, 'CDF', pd);
    result.ks_h = h3;
    result.ks_p = p3;
end