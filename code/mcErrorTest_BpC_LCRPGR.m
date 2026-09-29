function results = mcErrorTest_BpC_LCRPGR(POP, Builtup, varargin)
%% ============================================================
%  mcErrorTest_BpC_LCRPGR
%  Monte Carlo error test for BpC and LCRPGR
%
%  Usage:
%    results = mcErrorTest_BpC_LCRPGR(POP, Builtup)
%    results = mcErrorTest_BpC_LCRPGR(POP, Builtup, 'epsBuiltup', 0.10)
%    results = mcErrorTest_BpC_LCRPGR(POP, Builtup, ...
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
%       .BpC_true       - true BpC matrix (n x T)
%       .LCRPGR_true    - true LCRPGR matrix (n x T-1)
%       .BpC_sim        - BpC MC results (N x n x T)
%       .LCRPGR_sim     - LCRPGR MC results (N x n x T-1)
%       .BpC_stats      - BpC error statistics
%       .LCRPGR_stats   - LCRPGR error statistics
%       .BpC_ci         - BpC 95% confidence interval
%       .LCRPGR_ci      - LCRPGR 95% confidence interval
%       .BpC_normality  - BpC normality test results
%       .LCRPGR_normality - LCRPGR normality test results
%       .BpC_pdf        - BpC kernel density estimate
%       .LCRPGR_pdf     - LCRPGR kernel density estimate
%
%  Note:
%    epsBuiltup and epsPOP are INPUT relative errors.
%    BpC_stats.rel_std and LCRPGR_stats.rel_std are OUTPUT relative errors.
%    They are NOT the same quantity.
% ============================================================

%% 1. Parse input arguments
p = inputParser;
addRequired(p, 'POP', @(x) isnumeric(x) && ismatrix(x));
addRequired(p, 'Builtup', @(x) isnumeric(x) && ismatrix(x));
addParameter(p, 'epsBuiltup', 0.10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'epsPOP', 0.03, @(x) isnumeric(x) && x > 0);
addParameter(p, 'N', 10000, @(x) isnumeric(x) && x > 0 && mod(x,1)==0);
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

if T < 3
    error('The time series must contain at least 3 time points.');
end

% 检查非正值
if any(POP(:) <= 0) || any(Builtup(:) <= 0)
    warning('POP or Builtup contains non-positive values, replaced by 1e-6.');
    POP = max(POP, 1e-6);
    Builtup = max(Builtup, 1e-6);
end

%% 3. Compute true indicator values
BpC_true = Builtup ./ POP*1000000;              % n x T

LCRPGR_true = nan(n, T-1);              % n x (T-1)
for i = 1:n
    for t = 1:T-1
        num = log(Builtup(i, t+1) / Builtup(i, t));
        den = log(POP(i, t+1) / POP(i, t));
        if abs(den) > 1e-6
            LCRPGR_true(i, t) = num / den;
        else
            LCRPGR_true(i, t) = NaN;
        end
    end
end

%% 4. Monte Carlo simulation
fprintf('Starting Monte Carlo simulation, N = %d ...\n', N);

BpC_sim    = nan(N, n, T);
LCRPGR_sim = nan(N, n, T-1);

nan_LCRPGR_count = 0;

for k = 1:N
    % Add relative errors to each input variable
    % 使用截断正态分布，保证正值
    Builtup_s = Builtup .* max(1 + eps_B * randn(n, T), 1e-6);
    POP_s     = POP     .* max(1 + eps_P * randn(n, T), 1e-6);

    % BpC
    BpC_sim(k, :, :) = Builtup_s ./ POP_s *1000000; 
    BpC_sim2(:, :) = Builtup_s ./ POP_s *1000000; 
    % LCRPGR
    for i = 1:n
        for t = 1:T-1
            num = log(Builtup_s(i, t+1) / Builtup_s(i, t));
            den = log(POP_s(i, t+1) / POP_s(i, t));
            if abs(den) > 1e-6
                LCRPGR_sim(k, i, t) = num / den;
                LCRPGR_sim2(i, t) = num / den;
            else
                nan_LCRPGR_count = nan_LCRPGR_count + 1;
            end
        end
    end

    if mod(k, 1000) == 0
        fprintf('  Completed %d / %d\n', k, N);
    end
end

fprintf('Monte Carlo simulation finished.\n');
fprintf('LCRPGR NaN count = %d (%.4f%%)\n', ...
    nan_LCRPGR_count, 100 * nan_LCRPGR_count / (N * n * (T-1)));

%% 5. Error computation (using repmat to expand true values)
% BpC error
BpC_true_3d   = repmat(reshape(BpC_true,   [1, n, T]),   [N, 1, 1]);
BpC_err       = BpC_sim - BpC_true_3d;
BpC_err_vec   = BpC_err(:);
BpC_err_vec   = BpC_err_vec(~isnan(BpC_err_vec));

% LCRPGR error
LCRPGR_true_3d = repmat(reshape(LCRPGR_true, [1, n, T-1]), [N, 1, 1]);
LCRPGR_err     = LCRPGR_sim - LCRPGR_true_3d;
LCRPGR_err_vec = LCRPGR_err(:);
LCRPGR_err_vec = LCRPGR_err_vec(~isnan(LCRPGR_err_vec));

% Relative errors
% BpC 相对误差，剔除真值接近 0 的样本
valid_BpC = abs(BpC_true_3d) > 1e-6;
BpC_rel_err = nan(size(BpC_err));
BpC_rel_err(valid_BpC) = BpC_err(valid_BpC) ./ BpC_true_3d(valid_BpC);
BpC_rel_err_vec = BpC_rel_err(:);
BpC_rel_err_vec = BpC_rel_err_vec(~isnan(BpC_rel_err_vec));

% LCRPGR 相对误差，剔除真值接近 0 的样本
valid_LCRPGR = abs(LCRPGR_true_3d) > 1e-6;
LCRPGR_rel_err = nan(size(LCRPGR_err));
LCRPGR_rel_err(valid_LCRPGR) = LCRPGR_err(valid_LCRPGR) ./ LCRPGR_true_3d(valid_LCRPGR);
LCRPGR_rel_err_vec = LCRPGR_rel_err(:);
LCRPGR_rel_err_vec = LCRPGR_rel_err_vec(~isnan(LCRPGR_rel_err_vec));

%% 6. Statistics
BpC_stats = computeStats(BpC_err_vec, BpC_rel_err_vec);
LCRPGR_stats = computeStats(LCRPGR_err_vec, LCRPGR_rel_err_vec);

%% 7. Confidence intervals
BpC_ci = quantile(BpC_err_vec, [0.025, 0.5, 0.975]);
LCRPGR_ci = quantile(LCRPGR_err_vec, [0.025, 0.5, 0.975]);

BpC_rel_ci = quantile(BpC_rel_err_vec, [0.025, 0.5, 0.975]);
LCRPGR_rel_ci = quantile(LCRPGR_rel_err_vec, [0.025, 0.5, 0.975]);

%% 8. Normality tests
BpC_normality = normalityTest(BpC_err_vec);
LCRPGR_normality = normalityTest(LCRPGR_err_vec);

%% 9. Kernel density estimation
[BpC_pdf_f, BpC_pdf_x] = ksdensity(BpC_err_vec);
BpC_pdf.f = BpC_pdf_f;
BpC_pdf.x = BpC_pdf_x;

[LCRPGR_pdf_f, LCRPGR_pdf_x] = ksdensity(LCRPGR_err_vec);
LCRPGR_pdf.f = LCRPGR_pdf_f;
LCRPGR_pdf.x = LCRPGR_pdf_x;

%% 10. Package results
results = struct();
results.BpC_true        = BpC_true;
results.LCRPGR_true     = LCRPGR_true;
results.BpC_sim         = BpC_sim;
results.LCRPGR_sim      = LCRPGR_sim;
results.BpC_err_vec     = BpC_err_vec;
results.LCRPGR_err_vec  = LCRPGR_err_vec;
results.BpC_rel_err_vec = BpC_rel_err_vec;
results.LCRPGR_rel_err_vec = LCRPGR_rel_err_vec;
results.BpC_stats       = BpC_stats;
results.LCRPGR_stats    = LCRPGR_stats;
results.BpC_ci          = BpC_ci;
results.LCRPGR_ci       = LCRPGR_ci;
results.BpC_rel_ci      = BpC_rel_ci;
results.LCRPGR_rel_ci   = LCRPGR_rel_ci;
results.BpC_normality   = BpC_normality;
results.LCRPGR_normality = LCRPGR_normality;
results.BpC_pdf         = BpC_pdf;
results.LCRPGR_pdf      = LCRPGR_pdf;
results.epsBuiltup      = eps_B;
results.epsPOP          = eps_P;
results.N               = N;
results.nan_LCRPGR_count = nan_LCRPGR_count;

%% 11. Export results
if doExport
    if ~exist(exportDir, 'dir')
        mkdir(exportDir);
    end

    fid = fopen(fullfile(exportDir, 'mc_error_summary.txt'), 'w', 'n', 'UTF-8');
    fprintf(fid, '========================================\n');
    fprintf(fid, '  Monte Carlo Error Test for BpC and LCRPGR\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'Number of simulations N = %d\n', N);
    fprintf(fid, 'Relative error of built-up land epsBuiltup = %.4f\n', eps_B);
    fprintf(fid, 'Relative error of population      epsPOP     = %.4f\n\n', eps_P);

    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '  BpC error statistics\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean              = %.6f\n', BpC_stats.mean);
    fprintf(fid, 'Std               = %.6f\n', BpC_stats.std);
    fprintf(fid, 'Median            = %.6f\n', BpC_stats.median);
    fprintf(fid, 'MAD               = %.6f\n', BpC_stats.mad);
    fprintf(fid, 'Skewness          = %.6f\n', BpC_stats.skewness);
    fprintf(fid, 'Kurtosis          = %.6f\n', BpC_stats.kurtosis);
    fprintf(fid, 'Rel. mean         = %.6f\n', BpC_stats.rel_mean);
    fprintf(fid, 'Rel. std          = %.6f\n', BpC_stats.rel_std);
    fprintf(fid, '95%% CI            = [%.6f, %.6f]\n', BpC_ci(1), BpC_ci(3));
    fprintf(fid, 'Relative 95%% CI   = [%.6f, %.6f]\n', BpC_rel_ci(1), BpC_rel_ci(3));
    fprintf(fid, 'Lilliefors p      = %.6f\n', BpC_normality.lillie_p);
    fprintf(fid, 'Jarque-Bera p     = %.6f\n', BpC_normality.jb_p);
    if BpC_normality.lillie_p < 0.05
        fprintf(fid, 'Conclusion: BpC errors are NOT normally distributed\n');
    else
        fprintf(fid, 'Conclusion: BpC errors are normally distributed\n');
    end

    fprintf(fid, '\n----------------------------------------\n');
    fprintf(fid, '  LCRPGR error statistics\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean              = %.6f\n', LCRPGR_stats.mean);
    fprintf(fid, 'Std               = %.6f\n', LCRPGR_stats.std);
    fprintf(fid, 'Median            = %.6f\n', LCRPGR_stats.median);
    fprintf(fid, 'MAD               = %.6f\n', LCRPGR_stats.mad);
    fprintf(fid, 'Skewness          = %.6f\n', LCRPGR_stats.skewness);
    fprintf(fid, 'Kurtosis          = %.6f\n', LCRPGR_stats.kurtosis);
    fprintf(fid, 'Rel. mean         = %.6f\n', LCRPGR_stats.rel_mean);
    fprintf(fid, 'Rel. std          = %.6f\n', LCRPGR_stats.rel_std);
    fprintf(fid, '95%% CI            = [%.6f, %.6f]\n', LCRPGR_ci(1), LCRPGR_ci(3));
    fprintf(fid, 'Relative 95%% CI   = [%.6f, %.6f]\n', LCRPGR_rel_ci(1), LCRPGR_rel_ci(3));
    fprintf(fid, 'Lilliefors p      = %.6f\n', LCRPGR_normality.lillie_p);
    fprintf(fid, 'Jarque-Bera p     = %.6f\n', LCRPGR_normality.jb_p);
    if LCRPGR_normality.lillie_p < 0.05
        fprintf(fid, 'Conclusion: LCRPGR errors are NOT normally distributed\n');
    else
        fprintf(fid, 'Conclusion: LCRPGR errors are normally distributed\n');
    end

    fclose(fid);

    save(fullfile(exportDir, 'mc_error_results.mat'), 'results', '-v7.3');

    fprintf('Results exported to: %s\n', exportDir);
end

%% 12. Visualization
if doPlot
    figure('Position', [100, 100, 1400, 900], 'Color', 'w');

    subplot(2,3,1);
   
    BpC_err_vec_plot = BpC_err_vec(BpC_err_vec >= -50 & BpC_err_vec <= 50);
    histogram(BpC_err_vec, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
    plot(BpC_pdf.x, BpC_pdf.f, 'r-', 'LineWidth', 2);
    xlabel('BpC absolute error');
    ylabel('Probability density');
    title('BpC error distribution');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-50, 50]);

    subplot(2,3,2);
    LCRPGR_err_vec_plot = LCRPGR_err_vec(LCRPGR_err_vec >= -50 & LCRPGR_err_vec <= 50);
    histogram(LCRPGR_err_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
    plot(LCRPGR_pdf.x, LCRPGR_pdf.f, 'r-', 'LineWidth', 2);
    xlabel('LCRPGR absolute error');
    ylabel('Probability density');
    title('LCRPGR error distribution');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-50, 50]);

    subplot(2,3,3);
    BpC_rel_err_vec_plot = BpC_rel_err_vec(BpC_rel_err_vec >= -1 & BpC_rel_err_vec <= 1);
    histogram(BpC_rel_err_vec, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    xlabel('BpC relative error');
    ylabel('Probability density');
    title('BpC relative error distribution');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-1, 1]);

    subplot(2,3,4);
    LCRPGR_rel_err_vec_plot = LCRPGR_rel_err_vec(LCRPGR_rel_err_vec >= -5 & LCRPGR_rel_err_vec <= 5);
    histogram(LCRPGR_rel_err_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    xlabel('LCRPGR relative error');
    ylabel('Probability density');
    title('LCRPGR relative error distribution');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-5, 5]);

    if doExport
        saveas(gcf, fullfile(exportDir, 'mc_error_distributions.png'));
    end
end

%% 13. Command-line output
fprintf('\n========================================\n');
fprintf('  BpC Monte Carlo error test results\n');
fprintf('========================================\n');
fprintf('Mean = %.6f, Std = %.6f\n', BpC_stats.mean, BpC_stats.std);
fprintf('Relative mean = %.6f, Std = %.6f\n', BpC_stats.rel_mean, BpC_stats.rel_std);
fprintf('95%% confidence interval = [%.6f, %.6f]\n', BpC_ci(1), BpC_ci(3));
fprintf('Normality test p = %.6f\n', BpC_normality.lillie_p);
if BpC_normality.lillie_p < 0.05
    fprintf('Conclusion: BpC errors are NOT normally distributed\n');
else
    fprintf('Conclusion: BpC errors are normally distributed\n');
end

fprintf('\n========================================\n');
fprintf('  LCRPGR Monte Carlo error test results\n');
fprintf('========================================\n');
fprintf('Mean = %.6f, Std = %.6f\n', LCRPGR_stats.mean, LCRPGR_stats.std);
fprintf('Relative mean = %.6f, Std = %.6f\n', LCRPGR_stats.rel_mean, LCRPGR_stats.rel_std);
fprintf('95%% confidence interval = [%.6f, %.6f]\n', LCRPGR_ci(1), LCRPGR_ci(3));
fprintf('Normality test p = %.6f\n', LCRPGR_normality.lillie_p);
if LCRPGR_normality.lillie_p < 0.05
    fprintf('Conclusion: LCRPGR errors are NOT normally distributed\n');
else
    fprintf('Conclusion: LCRPGR errors are normally distributed\n');
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