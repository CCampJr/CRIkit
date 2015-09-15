%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%     asLS - Asymmetric least square (asLS) baseline removal. 
%%%     
%%%     Compute the baseline_current of signal_input using an asymmetric 
%%%     least square methods (asLS, AsLS, ALS, etc) algorithm 
%%%     designed by P.H. Eilers and H.F.M. Boelens.
%%%
%%%
%%%     baseline = asLS_baseline_v1(signal, smoothness_param, min_diff)
%%%
%%%     Inputs:
%%%         signal - signal to find baseline
%%%         smoothness_param - smoothness parameter (default, 1e3)
%%%         min_diff - break iterations if difference is less than min_diff
%%%                     (default, 1e-6)
%%%     Outputs:
%%%         baseline - asLS baseline
%%%
%%%     CITATION:
%%%         P. H. C. Eilers, "A perfect smoother," Anal. Chem. 75,
%%%         3631-3636 (2003).
%%%
%%%         P. H. C. Eilers and H. F. M. Boelens, "Baseline correction with
%%%         asymmetric least squares smoothing," Unpublished. 
%%%         October 21, 2005. It may be available on the internet.
%%%         
%%%     NOTE:
%%%         This code is just an implementation directly from the work of
%%%         P. H. C. Eilers and H. F. M. Boelens (see CITATION). Please cite their work if
%%%         you use this asLS code. If you use this code for use in phase
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
%%%         v0: 8/1/2014
%%%         v1: 4/23/2015
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function baseline = asLS_baseline_v1(signal, smoothness_param, asym_param)
% Estimate baseline with asymmetric least squares
MIN_DIFF = 1e-6;
MAX_ITER = 100;
ORDER = 2;

signal_length = length(signal);
signal = signal(:);

%assert(rem(nargin-1,2) == 0,'Number of parameter pairs is in error');

if nargin == 1
    smoothness_param = 1e3;
    asym_param = 1e-4;
end

penalty_vector = ones(signal_length, 1);

difference_matrix = diff(speye(signal_length), ORDER);


if length(smoothness_param) == 1
    smoothness_param = smoothness_param*ones(signal_length,1);
else
    ;
end
smoothness_matrix = smoothness_param*ones(1,size(difference_matrix,1));

differ = zeros(MAX_ITER);

for count = 1:MAX_ITER
    Weights = spdiags(penalty_vector, 0, signal_length, signal_length);

    C = chol(Weights + (smoothness_matrix .* difference_matrix') * difference_matrix);
    
    if count > 1
        baseline_last = baseline;
    end
    
    baseline = C \ (C' \ (penalty_vector .* signal));

    % Test for convergence
    if count > 1
        differ(count) = sum(abs(baseline_last-baseline));
        if (sum(abs(baseline_last-baseline)) < MIN_DIFF)
            break;  % Change is negligible
        else
            ;
        end
    end
%     count
    penalty_vector = (asym_param) .* (signal > baseline) + (1-asym_param) .* (signal < baseline);
end
