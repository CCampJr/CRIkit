%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%     Demo - demonstration of phase retrieval and error correction 
%%%     
%%%     This demo uses the re-developed Hilbert-transform-based
%%%     Kramers-Kronig reltion to perform phase retrieval. It then performs
%%%     phase detrending using the asymmetric least squares method (and an
%%%     additional demonstration with the asymmetric penalized least
%%%     squares method). Scaling is performed using the built-in
%%%     Savitky-Golay filter.
%%%
%%%     NOTE: this is an un-optimized demo, just meant for demonstrative
%%%     purposed. Varying parameters (e.g., detrending params) will likely
%%%     produce better results.
%%%
%%%     Required additional functions: MyHilbert.m, asLS_baseline_v1.m,
%%%     arPLS_baseline_v0.m
%%%
%%%     Inputs:
%%%         None
%%%
%%%     CITATION: C. H. Camp Jr., Y. J. Lee, and M. T. Cicerone, 
%%%         "Quantitative, Comparable Coherent Anti-Stokes Raman Scattering
%%%         (CARS) Spectroscopy: Correcting Errors in Phase Retrieval,"
%%%         Journal of Raman Spectroscopy (2015). arXiv:1507.06543.
%%%
%%%     REFERENCES: 
%%%         Y. Liu, Y. J. Lee, and M. T. Cicerone, "Broadband CARS spectral
%%%         phase retrieval using a time-domain Kramers-Kronig transform,"
%%%         Optics Letters 34, 1363-1365 (2009).
%%%
%%%         P. H. C. Eilers, "A perfect smoother," Anal. Chem. 75,
%%%         3631-3636 (2003).
%%%
%%%         P. H. C. Eilers and H. F. M. Boelens, "Baseline correction with
%%%         asymmetric least squares smoothing," Unpublished. 
%%%         October 21, 2005. It may be available on the internet.
%%%
%%%         S.-J. Baek, A. Park, Y.-J. Ahn, and J. Choo, "Baseline 
%%%         correction using asymmetrically reweighted penalized least 
%%%         squares smoothing," Analyst 140, 250-257 (2015).
%%%
%%%     Charles H Camp Jr (charles.camp@nist.gov, ccampjr@gmail.com) 
%%%         v1_0: 9/15/2015
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WN = linspace(100,4000,1500);
WN = WN(:);

A = [1,1,1];
OMEGA = [1000, 2000, 3000];
GAMMA = [20, 20, 20];

CHI_R = zeros(size(WN));
CHI_NR = 0.055 + 0.*CHI_R;

for count = 1:length(A)
    CHI_R = CHI_R + A(count)./(OMEGA(count) - WN - j*GAMMA(count));
end

CHI = CHI_R + CHI_NR;

I_CARS = abs(CHI).^2;
I_NRB = abs(CHI_NR).^2;

%% Basic Retrieval

[KK_ideal_imag, KK_ideal_real] = KKHilbert(I_NRB, I_CARS);
KK_ideal = KK_ideal_real + j*KK_ideal_imag;
phase_ideal = phase(KK_ideal);

figure
plot(WN,I_CARS);
hold all
plot(WN, I_NRB);
legend('CARS','NRB');
xlabel('Wavenumber (cm^{-1})');
ylabel('Signal Int (au)');
title('CARS and NRB Signal (Spectra)');

figure
plot(WN, KK_ideal_imag)
hold all
plot(WN,imag(CHI_R)./abs(CHI_NR))
legend('Retrieved','Ideal');
xlabel('Wavenumber (cm^{-1})');
ylabel('Raman-Like Int. (no units)');
title('Raman-Like Spectra (Retrieved)');

%% Retrieval with surrogate NRB (REF)

I_REF = I_NRB.*(1.*exp(-(WN-1400).^2./(2000.^2)));
[KK_ref_imag, KK_ref_real] = KKHilbert(I_REF, I_CARS);

KK_ref = KK_ref_real + j*KK_ref_imag;
phase_ref = phase(KK_ref);

figure
plot(WN,I_REF)
hold all
plot(WN, I_NRB)
legend('Surrogate NRB', 'NRB');
xlabel('Wavenumber (cm^{-1})');
ylabel('Signal Int. (au)');
title('Ideal and Surrogate NRB Spectra');

figure
plot(WN,KK_ref_imag)
hold all
plot(WN,KK_ideal_imag)
legend('Retrieved (REF)', 'Retrieved (NRB, Ideal)')
xlabel('Wavenumber (cm^{-1})');
ylabel('Raman-Like Int. (no units)');
title('Raman-Like Spectra with Surrogate NRB Spectrum');

%% Phase-error correction (Note: 'ideal' now means ideal correction, 
%  not the actual ideal [i.e., without any error]
phase_error_ideal = phase_ref - phase_ideal;
phase_error_asls = asLS_baseline_v1(phase_ref, 1e2, 1e-6);
phase_error_arpls = arPLS_baseline_v0(phase_ref, 1e-1, 1e-6);

figure
subplot(2,1,1)
plot(WN, phase_ref);
hold all
plot(WN, phase_ideal);
legend('Phase (Ref)', 'Phase (NRB, Ideal)')
xlabel('Wavenumber (cm^{-1})');
ylabel('Phase (rad)');
title('Phase');

subplot(2,1,2)
plot(WN,phase_error_asls)
hold all
plot(WN,phase_error_arpls)
plot(WN, phase_error_ideal, 'k');
legend('Phase error (AsLS)', 'Phase error (arPLS)', 'Phase error (Ideal)')
xlabel('Wavenumber (cm^{-1})');
ylabel('Phase (rad)');
title('Phase Error');

% amplitude correction factors
amp_corr_ideal = exp(imag(Hilbert(phase_error_ideal)));
amp_corr_asls = exp(imag(Hilbert(phase_error_asls)));
amp_corr_arpls = exp(imag(Hilbert(phase_error_arpls)));

%% Scaling and final plots
scale_ideal = (KK_ideal_real./(amp_corr_ideal.*abs(KK_ref).*cos(phase_ref-phase_error_ideal)));
scale_asls = 1./sgolayfilt(amp_corr_asls.*abs(KK_ref).*cos(phase_ref-phase_error_asls),3,1401);
scale_arpls = 1./sgolayfilt(amp_corr_arpls.*abs(KK_ref).*cos(phase_ref-phase_error_arpls),3,1401);

figure
plot(WN, amp_corr_asls.*scale_asls.*abs(KK_ref).*sin(phase_ref - phase_error_asls))
hold all
plot(WN, amp_corr_arpls.*scale_arpls.*abs(KK_ref).*sin(phase_ref - phase_error_arpls))
plot(WN, amp_corr_ideal.*scale_ideal.*abs(KK_ref).*sin(phase_ref - phase_error_ideal),'r')
plot(WN, imag(CHI_R)./abs(CHI_NR),'k');

legend('Correction (AsLS)', 'Correction (arPLS)','Ideal Correction', 'Ideal')
xlabel('Wavenumber (cm^{-1})');
ylabel('Raman-Like Int. (no units)');
title('Raman-Like Spectra with Phase and Amplitude Error Correction');

%% Compare traditional Amplitude-detrending with the presented correction method
figure
subplot(2,1,1)
plot(WN, amp_corr_ideal.*scale_ideal.*abs(KK_ref).*sin(phase_ref - phase_error_ideal))
hold all
plot(WN, KK_ref_imag - arPLS_baseline_v0(KK_ref_imag, 1e-1, 1e-6),'r');
plot(WN, imag(CHI_R)./abs(CHI_NR),'k');

legend('Correction (Ideal)', 'Amplitude Detrending','Ideal')
xlabel('Wavenumber (cm^{-1})');
ylabel('Raman-Like Int. (no units)');
title('Correction: New Procedure vs Traditional Amplitude Detrending');

subplot(2,1,2)
plot(WN, imag(CHI_R)./abs(CHI_NR)-amp_corr_ideal.*scale_ideal.*abs(KK_ref).*sin(phase_ref - phase_error_ideal))
hold all
plot(WN, imag(CHI_R)./abs(CHI_NR)- (KK_ref_imag - arPLS_baseline_v0(KK_ref_imag, 1e-1, 1e-6)),'r');
xlabel('Wavenumber (cm^{-1})');
ylabel('Difference: Corr - Ideal (no units)');
legend('Correction (Ideal)', 'Amplitude Detrending')