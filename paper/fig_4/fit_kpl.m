%Setup HP toolbox
run /Users/blazeyt/Documents/MATLAB/hyperpolarized-mri-toolbox-1.6/startup.m;

%Load in data
pyr_data = importdata('single_echo_pyr_ud_mag_on_anat_c1_means.txt');
lac_data = importdata('single_echo_lac_ud_mag_on_anat_c1_means.txt');
n_t = size(pyr_data, 1);
met_norm = max(pyr_data);

%Define data array
S = zeros(2, n_t);
S(1, :) = pyr_data / met_norm;
S(2, :) = lac_data / met_norm;

%Define flip angles
flips = zeros(2, n_t);   
flips(1, :) = 0.18462105; 
flips(2, :) = 1.3620738; 

%Run fitting, fixing pyruvate T1
params_fixed.R1P = 1/45;
[p_hat, s_hat, u_hat, p_err] = fit_pyr_kinetics(S, 3, flips, params_fixed);

if (isOctave == 1)
    disp(sprintf('kPL: %f', p_hat.kPL))
    disp(sprintf('R1L: %f', p_hat.R1L))
else
    disp(sprintf('kPL: %f, Lower: %f, Upper: %f', p_hat.kPL, p_err.kPL.lb, p_err.kPL.ub))
    disp(sprintf('R1P: %f, Lower: %f, Upper: %f', p_hat.R1P, p_err.R1P.lb, p_err.R1P.ub))
    disp(sprintf('R1L: %f, Lower: %f, Upper: %f', p_hat.R1L, p_err.R1L.lb, p_err.R1L.ub))
endif


%Save results
save('kpl_fit.mat', 'S', 'p_hat', 's_hat', 'u_hat', 'p_err')

exit
