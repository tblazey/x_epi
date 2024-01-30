#!/bin/tcsh

#Define data inputs
set twix = meas_MID00656_FID232958_pulseq.dat
set ref = meas_MID00659_FID232961_pulseq.dat
set json = single_echo_seq.json
set ref_json = single_echo_ref.json
set out = single_echo
set force = 1

#Run recon script
if ( ! -e ${out}_pyr.nii.gz || $force == 1 ) then
   x_epi_recon $twix $json single_echo -mean -anat mag_1.nii.gz -n_rep 20 -ts 3 \
               -complex -ref_info $ref $ref_json -freq_off 8 8
   set force = 1
endif
if ($? == 1) exit

#Update partial fourier recon
if ( ! -e ${out}_pyr_cmplx_no_pocs.nii.gz || $force == 1 ) then
   mv ${out}_pyr_cmplx.nii.gz ${out}_pyr_cmplx_no_pocs.nii.gz
   python3 ../py/pocs_recon.py ${out}_pyr_cmplx_no_pocs.nii.gz 0.66 \
                               ${out}_pyr_cmplx -pf_axis 2 -plot -n_iter 10
   set force = 1
endif
if ($? == 1) exit

#Register intermediate anatomical to hi-res one
if ( ! -e mag_1_to_anat.mat || $force == 1 ) then
   flirt -in mag_1 -ref anat -dof 6 -omat mag_1_to_anat.mat -o mag_1_on_anat
   set force = 1
endif
if ($? == 1) exit

#Invert transformation
if ( ! -e anat_to_mag_1.mat || $force == 1 ) then
   convert_xfm -omat anat_to_mag_1.mat -inverse mag_1_to_anat.mat
   set force = 1
endif
if ($? == 1) exit

#Make mask for anatomcial
if ( ! -e anat_mask.nii.gz || $force == 1 ) then
    fslmaths anat -thr 50 -bin -s 5 -thr 0.5 -bin anat_mask 
    set force = 1
endif
if ($? == 1) exit

#Apply mask to magnitude
if ( ! -e mag_1_masked.nii.gz || $force == 1 ) then
   flirt -in anat_mask -ref mag_1 -applyxfm -init anat_to_mag_1.mat -o mag_1_mask
   fslmaths mag_1_mask -thr 0.5 -bin mag_1_mask
   fslmaths mag_1 -mas mag_1_mask mag_1_masked
   set force = 1
endif
if ($? == 1) exit

#Apply mask to anat
if ( ! -e anat_masked.nii.gz || $force == 1 ) then
   fslmaths anat -mas anat_mask anat_masked
   set force = 1
endif
if ($? == 1) exit

#Prep field map
if ( ! -e fmap.nii.gz || $force == 1 ) then
   fsl_prepare_fieldmap SIEMENS pha mag_1_masked fmap 2.46
   set force = 1
endif
if ($? == 1) exit

#Metabolite resampling
foreach met ( 'pyr' 'lac' )
   echo "Resampling $met"

   #Get transformation between metabolite and high res
   set xfm = ${out}_${met}_to_anat.mat
   if ( ! -e $xfm || $force == 1 ) then
      convert_xfm -omat $xfm -concat mag_1_to_anat.mat ${out}_${met}_to_mag_1.mat
      set force = 1
   endif
   if ($? == 1) exit
   
   #Invert transformation between magnitude and epi
   if ( ! -e mag_1_to_${out}_${met}.mat || $force == 1 ) then
      convert_xfm -omat mag_1_to_${out}_${met}.mat -inverse ${out}_${met}_to_mag_1.mat
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get shift map
   if ( ! -e ${out}_${met}_shift.nii.gz || $force == 1 ) then
   
      #Resample field map
      flirt -in fmap -ref ${out}_${met}_mean -applyxfm \
            -init mag_1_to_${out}_${met}.mat -o fmap_on_${out}_${met}
            
      #Scale to use for c13
      fslmaths fmap_on_${out}_${met} -div 3.9761 fmap_on_${out}_${met}
      
      #Convert to shift map
      if ( $met == "pyr" ) then
         set dwell = 0.00082
      else
         set dwell = 0.00066
      endif
      fugue --loadfmap=fmap_on_${out}_${met} --dwell=$dwell \
            --saveshift=${out}_${met}_shift
      
      set force = 1
           
   endif
   if ($? == 1) exit
   
   #Apply shift map
   if ( ! -e ${out}_${met}_ud_cmplx.nii.gz || $force == 1 ) then
      python3 ../py/shift_mod.py ${out}_${met}_cmplx.nii.gz ${out}_${met}_shift.nii.gz \
                                 ${out}_${met}_ud_cmplx
      set force = 1
   endif
   if ($? == 1) exit
                              
   #Get mean
   if ( ! -e ${out}_${met}_ud_mean_mag.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_mean.py ${out}_${met}_ud_cmplx.nii.gz \
                                  ${out}_${met}_ud_mean_mag 3  -mag
      set force = 1
   endif
   if ($? == 1) exit
   
   #Convert complex
   if ( ! -e ${out}_${met}_ud_mag.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_to_plr.py ${out}_${met}_ud_cmplx.nii.gz ${out}_${met}_ud
      set force = 1
   endif
   if ($? == 1) exit
   
   #Apply transformation
   set xfmd = ${out}_${met}_ud_mag_on_anat.nii.gz
   if ( ! -e $xfmd || $force == 1 ) then
      applywarp -i ${out}_${met}_ud_mag -r anat -o $xfmd --premat=$xfm --interp=spline
      set force = 1
   endif
   if ($? == 1) exit
   
   #Apply transformation to mean
   set mean_out = ${out}_${met}_ud_mean_mag_on_anat.nii.gz
   if ( ! -e $mean_out || $force == 1 ) then
      applywarp -i ${out}_${met}_ud_mean_mag -r anat -o $mean_out \
                --premat=$xfm --interp=spline
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get mask values for each image
   foreach mask ( c1 c2 )
      set roi_out = ${xfmd:r:r}_${mask}_means.txt
      if ( ! -e $roi_out || $force == 1 ) then
         fslstats -t $xfmd -k ${mask}_on_anat -M > $roi_out 
         set force = 1 
      endif
      
   end  
   if ($? == 1) exit
   
end

#Make figure
if ( ! -e fig_4.tiff || $force == 1 ) then
   python3 fig_4.py
endif
if ($? == 1) exit

#Get kpl. Use Matlab to get lower/upper bounds
if ( ! -e kpl_fit.mat || $force == 1 ) then
   octave --no-gui fit_kpl.m
endif
if ($? == 1) exit


