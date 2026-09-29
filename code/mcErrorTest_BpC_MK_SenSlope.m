function results = mcErrorTest_BpC_MK_SenSlope(POP, Builtup, varargin)
%% ============================================================
%  mcErrorTest_BpC_MK_SenSlope
%  Mann-Kendall trend analysis of BpC time series, followed by
%  Monte Carlo assessment of Sen's slope uncertainty.
%
%  Output statistics:
%    - Sen's slope absolute error: mean, std, 95% CI
%    - Sen's slope relative error: mean, std, 95% CI
%
%  Figures:
%    - Histogram of Sen's slope absolute error + KDE red line
%    - Histogram of Sen's slope relative error + KDE red line
%
%  Usage:
%    results = mcErrorTest_BpC_MK_SenSlope(POP, Builtup)
%    results = mcErrorTest_BpC_MK_SenSlope(POP, Builtup, 'epsBuiltup', 0.10)
%    results = mcErrorTest_BpC_MK_SenSlope(POP, Builtup, ...
%                'epsBuiltup', 0.10, 'epsPOP', 0.03, 'N', 1000)
%
%  Inputs:
%    POP      - n x T population matrix (rows = cities, cols = time points)
%    Builtup  - n x T built-up land matrix (same size as POP)
%
%  Optional name-value pairs:
%    'epsBuiltup' - relative error of built-up land, default 0.10
%    'epsPOP'     - relative error of population,  default 0.03
%    'N'          - number of Monte Carlo simulations, default 1000
%    'exportDir'  - output directory, default 'mc_error_results'
%    'export'     - whether to export results, default true
%    'plot'       - whether to plot, default true
%
%  Note:
%    - MK is applied per city. A city needs at least 4 valid time points.
%    - Valid time point: Builtup >= 0.1 AND POP >= 1000.
%    - epsBuiltup and epsPOP are INPUT relative errors.
% ============================================================

%% 1. Parse input arguments
p = inputParser;
addRequired(p, 'POP', @(x) isnumeric(x) && ismatrix(x));
addRequired(p, 'Builtup', @(x) isnumeric(x) && ismatrix(x));
addParameter(p, 'epsBuiltup', 0.10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'epsPOP', 0.03, @(x) isnumeric(x) && x > 0);
addParameter(p, 'N', 1000, @(x) isnumeric(x) && x > 0 && mod(x,1)==0);
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

if T < 4
    warning('T < 4, Mann-Kendall test may not be meaningful.');
end

if any(POP(:) <= 0) || any(Builtup(:) <= 0)
    warning('POP or Builtup contains non-positive values, replaced by 1e-6.');
    POP = max(POP, 1e-6);
    Builtup = max(Builtup, 1e-6);
end

%% 3. Valid time-point mask
% Builtup >= 0.1 且 POP >= 1000 的时间点参与 MK 分析
minBuiltup = 0.1;
minPOP     = 1000;

valid_point = false(n, T);
for i = 1:n
    for t = 1:T
        if Builtup(i, t) >= minBuiltup && POP(i, t) >= minPOP
            valid_point(i, t) = true;
        end
    end
end

fprintf('Valid time points (Builtup >= %.2f and POP >= %.0f): %d / %d (%.2f%%)\n', ...
    minBuiltup, minPOP, sum(valid_point(:)), n*T, ...
    100 * sum(valid_point(:)) / (n*T));

%% 4. True BpC and MK analysis
BpC_true = Builtup ./ POP * 1000000;    % n x T

SenSlope_true = nan(n, 1);

for i = 1:n
    ts = BpC_true(i, :);
    ts(~valid_point(i, :)) = NaN;
    ts = ts(~isnan(ts));

    if numel(ts) >= 4
        [~, ~, slope, ~] = mannKendall(ts);
    else
        slope = NaN;
    end

    SenSlope_true(i) = slope;
end

fprintf('True Sen''s slope: mean = %.6f, median = %.6f, std = %.6f\n', ...
    mean(SenSlope_true(~isnan(SenSlope_true))), ...
    median(SenSlope_true(~isnan(SenSlope_true))), ...
    std(SenSlope_true(~isnan(SenSlope_true))));

%% 5. Monte Carlo simulation
fprintf('Starting Monte Carlo simulation, N = %d ...\n', N);

SenSlope_sim = nan(N, n);

for k = 1:N
    % 对 Builtup 和 POP 施加相对误差扰动（截断正态，保证正值）
    Builtup_s = Builtup .* max(1 + eps_B * randn(n, T), 1e-6);
    POP_s     = POP     .* max(1 + eps_P * randn(n, T), 1e-6);

    % 模拟 BpC
    BpC_sim = Builtup_s ./ POP_s * 1000000;

    % 对每个城市做 MK，只保留 Sen's slope
    for i = 1:n
        ts = BpC_sim(i, :);
        ts(~valid_point(i, :)) = NaN;
        ts = ts(~isnan(ts));

        if numel(ts) >= 4
            [~, ~, slope, ~] = mannKendall(ts);
            SenSlope_sim(k, i) = slope;
        end
    end

    if mod(k, 100) == 0
        fprintf('  Completed %d / %d\n', k, N);
    end
end

fprintf('Monte Carlo simulation finished.\n');

%% 6. Sen's slope errors
% 绝对误差: SenSlope_sim - SenSlope_true
SenSlope_true_rep = repmat(SenSlope_true', [N, 1]);   % N x n
AbsErr = SenSlope_sim - SenSlope_true_rep;

% 相对误差: (sim - true) / true，剔除真值接近 0 的样本
valid_rel = abs(SenSlope_true_rep) > 1e-6;
RelErr = nan(size(AbsErr));
RelErr(valid_rel) = AbsErr(valid_rel) ./ SenSlope_true_rep(valid_rel);

% 展平并去 NaN
AbsErr_vec = AbsErr(:);
AbsErr_vec = AbsErr_vec(~isnan(AbsErr_vec));

RelErr_vec = RelErr(:);
RelErr_vec = RelErr_vec(~isnan(RelErr_vec));

%% 7. Statistics
SenSlope_abs_stats = struct();
SenSlope_abs_stats.mean = mean(AbsErr_vec);
SenSlope_abs_stats.std  = std(AbsErr_vec);
SenSlope_abs_stats.ci   = quantile(AbsErr_vec, [0.025, 0.5, 0.975]);

SenSlope_rel_stats = struct();
SenSlope_rel_stats.mean = mean(RelErr_vec);
SenSlope_rel_stats.std  = std(RelErr_vec);
SenSlope_rel_stats.ci   = quantile(RelErr_vec, [0.025, 0.5, 0.975]);

fprintf('\n----------------------------------------\n');
fprintf('  Sen''s slope absolute error\n');
fprintf('----------------------------------------\n');
fprintf('Mean   = %.6f\n', SenSlope_abs_stats.mean);
fprintf('Std    = %.6f\n', SenSlope_abs_stats.std);
fprintf('95%% CI = [%.6f, %.6f]\n', ...
    SenSlope_abs_stats.ci(1), SenSlope_abs_stats.ci(3));

fprintf('\n----------------------------------------\n');
fprintf('  Sen''s slope relative error\n');
fprintf('----------------------------------------\n');
fprintf('Mean   = %.6f\n', SenSlope_rel_stats.mean);
fprintf('Std    = %.6f\n', SenSlope_rel_stats.std);
fprintf('95%% CI = [%.6f, %.6f]\n', ...
    SenSlope_rel_stats.ci(1), SenSlope_rel_stats.ci(3));

%% 8. Kernel density estimation
[AbsErr_pdf_f, AbsErr_pdf_x] = ksdensity(AbsErr_vec);
AbsErr_pdf.f = AbsErr_pdf_f;
AbsErr_pdf.x = AbsErr_pdf_x;

[RelErr_pdf_f, RelErr_pdf_x] = ksdensity(RelErr_vec);
RelErr_pdf.f = RelErr_pdf_f;
RelErr_pdf.x = RelErr_pdf_x;

%% 9. Package results
results = struct();
results.BpC_true            = BpC_true;
results.SenSlope_true       = SenSlope_true;
results.SenSlope_sim        = SenSlope_sim;
results.AbsErr_vec          = AbsErr_vec;
results.RelErr_vec          = RelErr_vec;
results.AbsErr_pdf          = AbsErr_pdf;
results.RelErr_pdf          = RelErr_pdf;
results.SenSlope_abs_stats  = SenSlope_abs_stats;
results.SenSlope_rel_stats  = SenSlope_rel_stats;
results.valid_point         = valid_point;
results.epsBuiltup          = eps_B;
results.epsPOP              = eps_P;
results.N                   = N;

%% 10. Export results
if doExport
    if ~exist(exportDir, 'dir')
        mkdir(exportDir);
    end

    fid = fopen(fullfile(exportDir, 'mk_senslope_summary.txt'), 'w', 'n', 'UTF-8');
    fprintf(fid, '========================================\n');
    fprintf(fid, '  Mann-Kendall Sen''s Slope Uncertainty\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, 'Number of simulations N = %d\n', N);
    fprintf(fid, 'epsBuiltup = %.4f, epsPOP = %.4f\n', eps_B, eps_P);
    fprintf(fid, 'Valid time points = %d / %d (%.2f%%)\n\n', ...
        sum(valid_point(:)), n*T, 100 * sum(valid_point(:)) / (n*T));

    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '  Sen''s slope absolute error\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean   = %.6f\n', SenSlope_abs_stats.mean);
    fprintf(fid, 'Std    = %.6f\n', SenSlope_abs_stats.std);
    fprintf(fid, '95%% CI = [%.6f, %.6f]\n\n', ...
        SenSlope_abs_stats.ci(1), SenSlope_abs_stats.ci(3));

    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '  Sen''s slope relative error\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Mean   = %.6f\n', SenSlope_rel_stats.mean);
    fprintf(fid, 'Std    = %.6f\n', SenSlope_rel_stats.std);
    fprintf(fid, '95%% CI = [%.6f, %.6f]\n', ...
        SenSlope_rel_stats.ci(1), SenSlope_rel_stats.ci(3));

    fclose(fid);

    save(fullfile(exportDir, 'mk_senslope_results.mat'), 'results', '-v7.3');

    fprintf('Results exported to: %s\n', exportDir);
end

%% 11. Visualization (two histograms, each with a KDE red line)
if doPlot
    figure('Position', [100, 100, 1200, 500], 'Color', 'w');

    % 子图1：Sen's slope 绝对误差直方图 + KDE
    subplot(2,3,1);
    AbsErr_vec_plot = AbsErr_vec(AbsErr_vec >= -10 & AbsErr_vec <= 10);
    histogram(AbsErr_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
    plot(AbsErr_pdf.x, AbsErr_pdf.f, 'r-', 'LineWidth', 2);
    xlabel('Sen''s slope  absolute error');
    ylabel('Probability density');
    title('Sen''s slope  absolute error');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-10, 10]);

    % 子图2：Sen's slope 相对误差直方图 + KDE
    subplot(2,3,2);
    RelErr_vec_plot = RelErr_vec(RelErr_vec >= -1 & RelErr_vec <= 1);
    histogram(RelErr_vec_plot, 100, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.9]);
    hold on;
        xlabel('Sen''s slope relative error');
    ylabel('Probability density');
    title('Sen''s slope relative error');
    legend('Histogram', 'Kernel density');
    grid on;
    set(gca, 'Color', 'w');
    xlim([-1, 1]);

    if doExport
        saveas(gcf, fullfile(exportDir, 'mk_senslope_error_distributions.png'));
    end
end

end

%% ============================================================
%  Helper functions
% ============================================================

function  [S, Z, slope, H] = mannKendall(x)
% Mann-Kendall trend test + Sen's slope
% 只返回 slope：
%   若 H == 1，即显著上升趋势，返回 Sen's slope
%   否则返回 0
%
% 输入：
%   x     : 时间序列，行向量或列向量，不能含 NaN
%   alpha : 显著性水平，例如 0.05
%
% 输出：
%   slope : Sen's slope，仅当 H == 1 时有效，否则为 0
    alpha = 0.05;
    if nargin < 2 || isempty(alpha)
        alpha = 0.05;
    end

    x = x(:);
    n = numel(x);

    %% Mann-Kendall S statistic
    S = 0;
    for i = 1:n-1
        for j = i+1:n
            S = S + sign(x(j) - x(i));
        end
    end

    %% Variance of S, no ties assumed
    VarS = n * (n - 1) * (2 * n + 5) / 18;

    %% Z-score with continuity correction
    if S > 0
        Z = (S - 1) / sqrt(VarS);
    elseif S < 0
        Z = (S + 1) / sqrt(VarS);
    else
        Z = 0;
    end

    %% Two-sided p-value
    pval = 2 * (1 - normcdf(abs(Z)));

    %% Hypothesis test
    if pval <= alpha
        if Z > 0
            H = 1;      % significant increasing trend
        elseif Z < 0
            H = -1;     % significant decreasing trend
        else
            H = 0;
        end
    else
        H = 0;          % no significant trend
    end

    %% Sen's slope
    if H == 1
        slopes = [];
        for i = 1:n-1
            for j = i+1:n
                slopes(end+1) = (x(j) - x(i)) / (j - i); %#ok<AGROW>
            end
        end
        slope = median(slopes);
    else
        slope = 0;
    end
end