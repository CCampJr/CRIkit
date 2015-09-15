%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%     arPLS - Asymmetric reweighted penalized least square (arPLS) 
%%%     baseline removal. 
%%%     
%%%     Compute the baseline_current of signal_input using an asymmetric 
%%%     reweighted penalized least square methods (arPLS) algorithm 
%%%     designed by S.-J. Baek, et al. 
%%%
%%%
%%%     baseline = arPLS_baseline_v0(signal, smoothness_param, min_diff)
%%%
%%%     Inputs:
%%%         signal - signal to find baseline
%%%         smoothness_param - smoothness parameter (default, 1e3)
%%%         min_diff - break iterations if difference is less than min_diff
%%%                     (default, 1e-6)
%%%     Outputs:
%%%         baseline - arPLS baseline
%%%
%%%     CITATION:
%%%         S.-J. Baek, A. Park, Y.-J. Ahn, and J. Choo, "Baseline 
%%%         correction using asymmetrically reweighted penalized least 
%%%         squares smoothing," Analyst 140, 250-257 (2015).
%%%
%%%     NOTE:
%%%         This code is just an implementation directly from the work of
%%%         S.-J. Baek, et al. (see CITATION). Please cite their work if
%%%         you use this arPLS code. If you use this code for use in phase
%%%         retrieval and/or error correction for coherent Raman
%%%         spectroscopy/microscopy, please cite our work (see APPLICATION
%%%         REFERENCE).
%%%
%%%     APPLICATION REFERENCE: 
%%%         C. H. Camp Jr., Y. J. Lee, and M. T. Cicerone, "Quantitative, 
%%%         Comparable Coherent Anti-Stokes Raman Scattering (CARS) 
%%%         Spectroscopy: Correcting Errors in Phase Retrieval,"
%%%         Journal of Raman Spectroscopy (2015). arXiv:1507.06543.
%%%
%%%     Charles H. Camp Jr (charles.camp@nist.gov, ccampjr@gmail.com) 
%%%         v0: 9/15/2015
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function baseline = arPLS_baseline_v0(signal, smoothness_param, min_diff)

ORDER = 2; % Difference filter order
MAX_ITER = 100; % Maximum iterations

[m,n] = size(signal);
if (m ~= 1 && n ~= 1)
    error('This function only accepts 1D (effective) signals');
end
signal = signal(:);

if nargin == 1
    smoothness_param = 1e3;
    min_diff = 1e-6;
end

signal_length = length(signal);
difference_matrix = diff(speye(signal_length), ORDER);
minimization_matrix = (smoothness_param*difference_matrix')*difference_matrix;
penalty_vector = ones(signal_length,1);

for count = 1:MAX_ITER
    penalty_matrix = spdiags(penalty_vector, 0, signal_length, signal_length);
    % Cholesky decomposition
    C = chol(penalty_matrix + minimization_matrix);
    baseline = C \ (C'\(penalty_vector.*signal));
    d = signal - baseline;
    % make d-, and get penalty_vector^t with m and s
    dn = d(d<0);
    m = mean(dn);
    s = std(dn);
    penalty_vector_temp = 1./(1+exp(2*(d-(2*s-m))/s));
    % check exit condition and backup
    if norm(penalty_vector-penalty_vector_temp)/norm(penalty_vector) < min_diff
%         count
        break;
    end
    penalty_vector = penalty_vector_temp;
end

