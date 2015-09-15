%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%     Hilbert - FFT implementation of the Hilbert transform that takes
%%%     in a signal (or multiple signals in parallel) and outputs an
%%%     analytic signal(s) based on the Hilbert transform.
%%%
%%%     If you use this software, please cite our work (see below).
%%%
%%%     Analytic = Hilbert(Input,padfactor)
%%%
%%%     Inputs:
%%%         Input - input signal(s) (M x 1; 1 x M; M x N) -- see note on
%%%             parallel processing
%%%         padfactor (optional; default = 1) - (0 or positive integer) multiplier of the original
%%%             signal size M for each padding region. E.g., M = 1000,
%%%             padfactor = X, length of signal during processing X*M+M+X*M =
%%%             2X*M+M. Output signal length = M.
%%%     Output:
%%%         Analytic - analytical representation of input signal. NOTE: real(Analytic) =
%%%         Input. imag(Analytic) = Hilbert{Input}
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
%%%     Charles H Camp Jr (charles.camp@nist.gov, ccampjr@gmail.com)
%%%         v0: 9/11/2014
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Analytic = Hilbert(Input,varargin)

% Was padfactor given?
if length(varargin) == 1
    pad_factor = varargin{1};
else
    pad_factor = 1;
end

% Ensure a 1D signal is M x 1; else make it M x 1.
[m,n] = size(Input);
if (m == 1 | n == 1)
    Input = Input(:);
else
    ;
end
[m,n] = size(Input);

% Add padfactor*M padding. Before -- padfactor*M @ Amp = Input(1); After -- padfactor*M @ Amp = Input(end);
pad_low = ones(pad_factor*m,1)*Input(1,:);
pad_high = ones(pad_factor*m,1)*Input(end,:);

Input_padded = [pad_low;Input;pad_high];
[m2,n2] = size(Input_padded);

% Operate in Fourier-domain
Input_fft_padded = fft(Input_padded);
t_temp = linspace(1,-1,m2).';
Mask = j.*sign(t_temp); % Represents Hilbert as an amplitude mask in the Fourier-domain
%Transform_mask = Mask*ones(1,n2);
Transform_mask = t_temp*ones(1,n2);
Analytic_fft_padded = Transform_mask.*Input_fft_padded;

% iFFT
%Analytic_padded = real(ifft(Analytic_fft_padded));
Analytic_padded = ifft(Analytic_fft_padded);

% De-pad and output analytic signal
%Analytic = Input + j.*(Analytic_padded(m*pad_factor+1:m*pad_factor+m,:));
Analytic = Input + j.*imag(Analytic_padded(m*pad_factor+1:m*pad_factor+m,:));


