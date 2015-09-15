%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%     KKHilbert - Retrieve real and imaginary components of raw CARS
%%%     spectrum utilizing a Kramers-Kronig relation. 
%%%     
%%%     This is a re-implementation of the "modified time-domain 
%%%     Kramers-Kronig transform" (see References) that is computationally
%%%     more efficient and utilizes the Hilbert transform explicitly. If
%%%     you use this software please cite our work (see below).
%%%
%%%     Required additional functions: MyHilbert.m
%%%
%%%     [KK_Imag,KK_Real] = KKHilbert(NRB,CARS,[PhaseOffset],[NRBNorm])
%%%
%%%     Inputs:
%%%         NRB - Nonresonant background signal(s). (M x 1; 1 x M; or M x N) 
%%%             See note on parallel computations
%%%         CARS - Raw BCARS signal(s). (M x 1; 1 x M; or M x N). 
%%%             See note on parallel computations.
%%%         PhaseOffset (optional, default = 0)- (rad). (1 x 1; M x 1; 1 x M; or M x N).
%%%             Constant or vector phase to add to retrieved phase of KKd and KKdR
%%%         NRBNorm (optional, default = 1 [Yes]) - (1 x 1). Normalize 
%%%             Retrieved spectrum to the NRB: 1 = yes, 0 = No.
%%%
%%%         NOTE ON OPTIONAL PARAMETERS - Neither or both optional
%%%             parameters must be provided.
%%%
%%%     Outputs:
%%%         KK_Imag - imaginary component of input signal(s). (M x 1, or M x N)
%%%         KK_Real - real component of input signal(s). (M x 1, or M x N)
%%%
%%%     NOTE ON PARALLEL PROCESSING - This function can process N spectra
%%%     simultaneously. Make sure the input signal is M x N, with M =
%%%     signal length, N = # of signals
%%%
%%%     CITATION: C. H. Camp Jr., Y. J. Lee, and M. T. Cicerone, 
%%%         "Quantitative, Comparable Coherent Anti-Stokes Raman Scattering
%%%         (CARS) Spectroscopy: Correcting Errors in Phase Retrieval,"
%%%         Journal of Raman Spectroscopy (2015). arXiv:1507.06543.
%%%
%%%     REFERENCE: Y. Liu, Y. J. Lee, and M. T. Cicerone, 
%%%         "Broadband CARS spectral phase retrieval using a time-domain 
%%%         Kramers-Kronig transform," Optics Letters 34, 1363-1365 (2009).
%%%
%%%     Charles H Camp Jr (charles.camp@nist.gov, ccampjr@gmail.com) 
%%%         v2_0: 9/11/2014
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [KK_Imag,KK_Real] = KKHilbert(NRB,CARS,PhaseOffset,NRBNorm)

% Are optional parameters supplied?
if nargin == 2 % No. Set defaults
    PhaseOffset = 0;
    NRBNorm = 1;
end

% Ensure 1D PhaseOfsset size is M x 1
if(size(PhaseOffset,1) == 1 | size(PhaseOffset,2) == 1)
    PhaseOffset = PhaseOffset(:);
else
    ;
end

[m,n] = size(CARS);
[m2,n2] = size(NRB);

% If CARS is 1D, ensure M x 1 size
if m == 1 & n > 1
    CARS = CARS.';
    [m,n] = size(CARS);
end

% If NRB is 1D, ensure M x 1 size
if m2 == 1 & n2 > 1
    NRB = NRB.';
    [m2,n2] = size(NRB);
end

% Make sure NRB and CARS spectral length are the same
if m~= m2
    errordlg(['Incorrect Spectral Axis Size. NRB: ' num2str(m2) ' CARS: ' num2str(m)])
    error(['Incorrect Spectral Axis Size. NRB: ' num2str(m2) ' CARS: ' num2str(m)])
end

% For N > 1 signals, make sure NRB has either 1 spectrum (applied across
% all CARS spectra) or N spectra (N = # CARS spectra)
if (n2 > 1 & n2 ~= n)
    errordlg(['Incorrect # of Spectra. NRB: ' num2str(n2) ' CARS: ' num2str(n)])
    error(['Incorrect # of Spectra. NRB: ' num2str(n2) ' CARS: ' num2str(n)])
elseif (n2 == 1 & n > 1) % If CARS = N spectra and NRB = 1 spectrum; Make N replicates of NRB
    NRB = NRB*ones(1,n);
end

temp_in = CARS./NRB;

% Find and replace NaN and Inf values with 1 (log(1) = 0). Alterantive, can
% replace 1 with a small value, but may greatly increase noise.
vec = find(isnan(temp_in)==1);
temp_in(vec) = 1;
vec = find(isinf(temp_in)==1);
temp_in(vec) = 1;
vec = find(temp_in==0);
temp_in(vec) = 1;


if NRBNorm == 1 % Normalize to NRB -- (Output = sqrt(CARS/NRB)*exp(j*phi))
    temp = real(sqrt(temp_in)).*exp(j.*PhaseOffset+j.*imag(Hilbert(.5.*real(log(temp_in)))));
else % DO NOT Normalize to NRB -- (Output = sqrt(CARS)*exp(j*phi))
    temp = real(sqrt(CARS)).*exp(j.*PhaseOffset+j.*imag(Hilbert(.5.*real(log(temp_in)))));
end

KK_Imag = imag(temp);
KK_Real = real(temp);