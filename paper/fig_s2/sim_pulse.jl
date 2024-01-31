#Load packages
using DelimitedFiles
using FFTW
using FiniteDiff
using LinearAlgebra
using NPZ
using ProgressMeter

#Constants and such
rf_name = "./siemens_singleband_pyr_3T.RF";
grd_name = "./siemens_singleband_pyr_3T.GRD";
out_name = "./pulse_sim.npy";
grd_max = 1.3698;                                            #Gauss / cm
b1_max = 0.1176;                                             #Gauss
gamma = 10.7084E6;                                           #Hz/T
freq_offs = -600:1.2:600;                                    #Hz
dists = -24:0.048:24;                                        #cm
dt = 4E-6;                                                   #s

#Unit conversion
grd_max *= 1E-4 * 1E2 * gamma * 2 * pi;                      #rad / m / s
b1_max *= 1E-4 * gamma * 2 * pi;                             #rad / s 
freq_offs *= 2 * pi;                                         #rad / s
dists /= 100;                                                #m                                  

#Define spin bases
spin_x = ComplexF64[0 0.5; 0.5 0];
spin_y = ComplexF64[0 -0.5im; 0.5im 0];
spin_z = ComplexF64[0.5 0; 0 -0.5];

#Load in rf shapes
rf = readdlm(rf_name, comments=true);
grd = readdlm(grd_name, comments=true);

#Counts
n_t = size(rf)[1];
n_f = size(freq_offs)[1];
n_d = size(dists)[1];

#Extract mag and phase
pha = rf[:, 1] / 180 * pi;                                  #radians
mag = rf[:, 2];
mag *= b1_max / maximum(abs.(mag));                         #rad / s

#Get frequency from phase
freq = [0; diff(pha) / dt];                                 #rad / s

#Convert gradient
grd *= grd_max / maximum(abs.(grd));                        #rad / m / s

#Make empty array for storing signal                      
sig = zeros(ComplexF64, n_f, n_d);

#Loop through frequency offsets
@showprogress for (i, freq_off) in enumerate(freq_offs)

   #Loop through distances
   for (j, dist) in enumerate(dists)
   
      #Initialize spin system
      spin = spin_z;
   
      #Loop through timepoint of pulses
      for k = 1:n_t
      
         #Update spin system
         ham_z = spin_z * (freq_off + freq[k] + grd[k] * dist);
         ham_x = spin_x * mag[k];
         ham = ham_z + ham_x;
         arg = exp(-1im * ham * dt);
         spin = arg * spin * arg';
      
      end
      
      #Get signal in transverse plane as a complex number
      sig[i, j] = real(tr(spin_x * spin') / 0.5) + 1im * real(tr(spin_y * spin') / 0.5);
   
   end
   
end

#Save data
npzwrite(out_name, Dict("sig" => sig, "freqs" => freq_offs, "dists" => dists));

