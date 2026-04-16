% function [A, x0, y, meta] = gen_binary_cs_instance(N, alpha, rho, mu, varargin)
% % GEN_BINARY_CS_INSTANCE  Generate one binary compressed sensing instance
% % consistent with the JPSJ'24 paper’s Section 2 settings.
% %
% % Model:
% %   x0 ~ Bernoulli(rho) in {0,1}^N          (paper Eq. (4))
% %   A_ij ~ N( mu/N, 1/N )                   (paper Eq. (5): var = 1/N, nonzero mean)
% %   y = A*x0  (+ optional Gaussian noise)
% %
% % Usage:
% %   [A,x0,y,meta] = gen_binary_cs_instance(500, 0.4, 0.2, 2, 'noise_sigma',0, 'seed',1);
% %
% % Inputs:
% %   N      : signal dimension
% %   alpha  : compression ratio M/N  (M = round(alpha*N))
% %   rho    : nonzero rate in x0  (Pr{x0_i=1} = rho)
% %   mu     : bias level for A’s mean (use {0,2,10} to mirror the paper)
% %
% % Name-Value options:
% %   'noise_sigma' : std of Gaussian noise added to y (default 0 = noiseless)
% %   'seed'        : RNG seed for reproducibility (default [])
% %
% % Outputs:
% %   A   : M-by-N sensing matrix
% %   x0  : N-by-1 ground-truth binary vector in {0,1}
% %   y   : M-by-1 measurements (A*x0 plus optional noise)
% %   meta: struct with fields N, M, alpha, rho, mu, noise_sigma, seed
% 
% % ---- parse options
% p = inputParser;
% addParameter(p,'noise_sigma', 0);
% addParameter(p,'seed', []);
% parse(p, varargin{:});
% noise_sigma = p.Results.noise_sigma;
% seed        = p.Results.seed;
% 
% % ---- sizes
% M = max(1, round(alpha*N));
% 
% % ---- RNG
% if ~isempty(seed)
%  
% randn('state', double(seed));
% rand('state', double(seed)); 
% end
% 
% % ---- ground-truth signal x0 ~ Bernoulli(rho) in {0,1}
% x0 = double(rand(N,1) < rho);
% 
% % ---- sensing matrix A with bias and variance matching paper:
% % A_ij ~ N(mu/N, 1/N)
% biasMean = mu / N;
% biasStd  = 1 / sqrt(N);
% A = biasMean + biasStd * randn(M, N);
% 
% % ---- measurements
% y = A * x0;
% 
% % ---- optional additive Gaussian noise
% if noise_sigma > 0
%     y = y + noise_sigma * randn(M,1);
% end
% 
% % ---- metadata
% meta = struct('N',N,'M',M,'alpha',alpha,'rho',rho,'mu',mu, ...
%               'noise_sigma',noise_sigma,'seed',seed);
% end

function [A, x0, y, meta] = gen_binary_cs_instance(N, alpha, rho, mu, varargin)
% GEN_BINARY_CS_INSTANCE  Generate one binary compressed sensing instance
% with calibration aids for very-sparse cases.
%
% Model (base):
%   x0 ~ Bernoulli(rho) in {0,1}^N         (or exactly-k sparse if 'k' given)
%   A_ij ~ N(mu/N, 1/N)
%   y = A*x0  (+ optional Gaussian noise)
%
% Extras (for DCRA and hard sparse regimes):
%   - Column normalize A
%   - Provide Atil=0.5*A, btil=y-0.5*A*1, cz=(lambda/2)*1
%   - Provide augmented (Abar,bbar) for f(Az-b) form
%   - Heuristic lambda0 and a continuation schedule
%
% Usage (same signature as before):
%   [A,x0,y,meta] = gen_binary_cs_instance(500, 0.4, 0.2, 2, 'noise_sigma',0, 'seed',1);
%
% Name-Value options:
%   'noise_sigma'    : std of Gaussian noise added to y (default 0)
%   'seed'           : RNG seed (default [])
%   'normalize_cols' : true/false, normalize columns of A (default true)
%   'k'              : exact sparsity (integer). If provided, ignore Bernoulli rho.
%   'lambda'         : regularization used to build cz (default: auto via lambda0)
%   'lambda_schedule': custom row vector of lambdas for continuation (default auto)

% ---- parse options
p = inputParser;
addParameter(p,'noise_sigma', 0);
addParameter(p,'seed', []);
addParameter(p,'normalize_cols', false);
addParameter(p,'k', []);                  % exact # of ones (optional)
addParameter(p,'lambda', []);             % if empty, compute lambda0
addParameter(p,'lambda_schedule', []);    % if empty, generate gentle schedule
parse(p, varargin{:});
noise_sigma     = p.Results.noise_sigma;
seed            = p.Results.seed;
normalize_cols  = p.Results.normalize_cols;
k_exact         = p.Results.k;
lambda_user     = p.Results.lambda;
lambda_sched_in = p.Results.lambda_schedule;

% ---- sizes
M = max(1, round(alpha*N));

% ---- RNG (recommended modern interface)
if ~isempty(seed)
    rng(double(seed), 'twister');
end

% ---- ground-truth signal x0
if ~isempty(k_exact)
    % exact-k sparsity (robust for tiny rho)
    x0 = zeros(N,1);
    idx = randperm(N, min(k_exact, N));
    x0(idx) = 1;
else
    % Bernoulli(rho)
    x0 = double(rand(N,1) < rho);
end

% ---- sensing matrix A with bias and variance matching paper:
% A_ij ~ N(mu/N, 1/N)
biasMean = mu / N;
biasStd  = 1 / sqrt(N);
A = biasMean + biasStd * randn(M, N);

% ---- optional column normalization (stabilizes very-sparse recovery)
if normalize_cols
    colnorms = vecnorm(A, 2, 1);
    colnorms = max(colnorms, 1e-12);     % avoid divide-by-zero
    A = A * spdiags(1./colnorms', 0, N, N);
end

% ---- measurements
y = A * x0;

% ---- optional additive Gaussian noise
if noise_sigma > 0
    y = y + noise_sigma * randn(M,1);
end

% ---- Build ±1 reformulation helpers and calibration artefacts
% tilde (±1) pair
Atil = 0.5 * A;
btil = y - 0.5 * (A * ones(N,1));

% heuristic lambda_0 if not provided:
% robust scale using a correlation statistic
if isempty(lambda_user)
    % Use median absolute correlation magnitude as scale proxy
    sgn_res = sign(Atil * sign(Atil' * btil + 1e-12) - btil);
    lam0 = 0.5 * median(abs(Atil' * sgn_res));
    if ~isfinite(lam0) || lam0 <= 0
        lam0 = 0.05;  % safe fallback
    end
    lambda = lam0;
else
    lambda = lambda_user;
end
cz = (lambda/2) * ones(N,1);

% augmented trick packs for f(Az - b) interface
Abar = [Atil; speye(N)];
bbar = [btil; zeros(N,1)];
M_resid = size(Atil,1);   % first-block length for prox/envelope

% default gentle lambda schedule if not given (helps very-sparse)
if isempty(lambda_sched_in)
    lambda_schedule = lambda * (1.25) .^ (0:6);   % 7-stage growth
else
    lambda_schedule = lambda_sched_in(:).';
end

% ---- metadata
meta = struct();
meta.N          = N;
meta.M          = M;
meta.alpha      = alpha;
meta.rho        = rho;
meta.mu         = mu;
meta.k_exact    = k_exact;
meta.normalize_cols = normalize_cols;
meta.noise_sigma= noise_sigma;
meta.seed       = seed;

% ±1 reformulation and augmented forms
meta.Atil       = Atil;
meta.btil       = btil;
meta.cz         = cz;
meta.M_resid    = M_resid;
meta.Abar       = Abar;
meta.bbar       = bbar;

% lambda calibration
meta.lambda     = lambda;
meta.lambda_schedule = lambda_schedule;

% for reproducibility / diagnostics
meta.colnorms   = exist('colnorms','var') * (exist('colnorms','var') ~= 0);
if meta.colnorms
    meta.colnorms = colnorms;  %#ok<NASGU> % (kept for debugging; comment out if large)
end
end

