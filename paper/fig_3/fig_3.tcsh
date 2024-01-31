#!/bin/tcsh -f
set force = 1
#Run recon
if ( $#argv == 1 ) then
    set force = $argv
else
    set force = 0
endif

if ( ! -e multi_echo_eg.nii.gz || $force == 1 ) then
   x_epi_recon meas_MID00123_FID231462_pulseq.dat multi_echo_no_rs.json multi_echo \
               -n_avg 32 -anat ../fig_2/mag.nii.gz -complex \
               -ref_info meas_MID00127_FID231466_pulseq.dat multi_echo_no_rs_ref.json
   set force = 1
endif
if ($? == 1) exit

#Update partial fourier recon
if ( ! -e multi_echo_eg_cmplx_no_pocs.nii.gz || $force == 1 ) then
   mv multi_echo_eg_cmplx.nii.gz multi_echo_eg_cmplx_no_pocs.nii.gz
   python3 ../py/pocs_recon.py multi_echo_eg_cmplx_no_pocs.nii.gz 0.75 \
                               multi_echo_eg_cmplx -pf_axis 2 -plot -n_iter 15 \
                               -skip_half upper   
   set force = 1
endif
if ($? == 1) exit

#Loop through metabolites
foreach met ( eg me )
   
   #Apply distortion correction
   if ( ! -e multi_echo_${met}_ud_cmplx.nii.gz || $force == 1 ) then
      python3 ../py/shift_mod.py multi_echo_${met}_cmplx.nii.gz \
                                 ../fig_2/single_echo_${met}_shift.nii.gz \
                                 multi_echo_${met}_ud_cmplx
      set force = 1
   endif
   if ($? == 1) exit
   
   #Convert distortion corrected image to magnitude
   if ( ! -e multi_echo_${met}_ud_mag.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_to_plr.py multi_echo_${met}_ud_cmplx.nii.gz multi_echo_${met}_ud
   endif
   if ($? == 1) exit
   
   #Compute mean across time and echoes
   if ( ! -e multi_echo_${met}_ud_echo_mean.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_mean.py multi_echo_${met}_ud_cmplx.nii.gz \
                                  multi_echo_${met}_ud_echo_mean 4 -mag
                                  
      set force = 1
   endif
   if ($? == 1) exit
   
   #Compute mean across time and echoes
   if ( ! -e multi_echo_${met}_ud_echo_sum.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_mean.py multi_echo_${met}_ud_cmplx.nii.gz \
                                  multi_echo_${met}_ud_echo_sum 4 -mag -sum
                                  
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get transformations between metabolite images and anatomical
   if ( ! -e multi_echo_${met}_to_anat.mat || $force == 1 ) then
      convert_xfm -concat ../fig_2/mag_to_anat.mat \
                          multi_echo_${met}_to_mag.mat \
                  -omat multi_echo_${met}_to_anat.mat
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get snr image for each echo
   if ( ! -e multi_echo_${met}_ud_snr.nii.gz || $force == 1 ) then
      python3 ../py/compute_snr_vox.py multi_echo_${met}_ud_mag.nii.gz \
                                       multi_echo_${met}_ud_snr
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get snr image for average across echoes
   foreach op ( mean sum )
      if ( ! -e multi_echo_${met}_ud_echo_${op}_snr.nii.gz || $force == 1 ) then
         python3 ../py/compute_snr_vox.py multi_echo_${met}_ud_echo_${op}.nii.gz \
                                          multi_echo_${met}_ud_echo_${op}_snr
         set force = 1
      endif
   end
   if ($? == 1) exit
   
   #Resample epi images to anatomical
   foreach suff ( snr echo_mean_snr echo_sum_snr mag )
      
      set img_in = multi_echo_${met}_ud_${suff}
      if ( ! -e ${img_in}_on_anat.nii.gz || $force == 1 ) then
         set n_echo = `fslval $img_in dim5`
         if ( $n_echo == 1 ) then
            flirt -in $img_in -ref ../fig_2/anat -applyxfm -interp spline \
                  -init multi_echo_${met}_to_anat.mat -o ${img_in}_on_anat.nii.gz
         else
          
            #Swap dims
            python3 ../py/swap_dims.py ${img_in}.nii.gz ${img_in}_swap 0 1 2 4 3
            
            #Apply transformation
            flirt -in ${img_in}_swap -ref ../fig_2/anat -applyxfm -interp spline \
                  -init multi_echo_${met}_to_anat.mat -o ${img_in}_swap_on_anat.nii.gz
                  
            #Swap back
            python3 ../py/swap_dims.py ${img_in}_swap_on_anat.nii.gz ${img_in}_on_anat \
                                       0 1 2 4 3 -add 1
          
         endif
         
         set force = 1
      endif
      if ($? == 1) exit
   
   end
   
end

#Run python script to make figure
if ( ! -e fig_3.tiff || $force == 1 ) then
   python3 fig_3.py
endif

