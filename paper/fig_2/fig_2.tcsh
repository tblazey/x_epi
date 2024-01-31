#!/bin/tcsh -f
set force = 1

#Run recon
if ( ! -e single_echo_eg_cmplx.nii.gz || $force == 1 ) then
   x_epi_recon meas_MID00120_FID231459_pulseq.dat \
               single_echo_no_rs.json \
               single_echo -n_avg 32 -anat mag.nii.gz \
               -ref_info meas_MID00124_FID231463_pulseq.dat single_echo_no_rs_ref.json \
               -complex
   set force = 1
endif
if ($? == 1) exit

#Update partial fourier recon
if ( ! -e single_echo_eg_cmplx_no_pocs.nii.gz || $force == 1 ) then
   mv single_echo_eg_cmplx.nii.gz single_echo_eg_cmplx_no_pocs.nii.gz
   python3 ../py/pocs_recon.py single_echo_eg_cmplx_no_pocs.nii.gz 0.66 \
                               single_echo_eg_cmplx -pf_axis 2 -plot -n_iter 15 \
                               -skip_half upper   
   set force = 1
endif
if ($? == 1) exit
                    
#Register mag to anatomical
if ( ! -e mag_to_anat.mat || $force == 1 ) then
   flirt -in mag -ref anat -dof 6 -omat mag_to_anat.mat \
         -o mag_on_anat
   convert_xfm -omat anat_to_mag.mat -inverse mag_to_anat.mat
   set force = 1
endif
if ($? == 1) exit

#Make mask for anatomcial
if ( ! -e anat_mask.nii.gz || $force == 1 ) then
    fslmaths anat -thr 50 -bin -s 5 -thr 0.5 -bin anat_mask 
    set force = 1
endif
if ($? == 1) exit

#Get magnitude mask
if ( ! -e mag_mask.nii.gz || $force == 1 ) then
   flirt -in anat_mask -ref mag -applyxfm -init anat_to_mag.mat -o mag_mask
   fslmaths mag_mask -thr 0.5 -bin mag_mask
   fslmaths mag -mas mag_mask mag_masked
   set force = 1
endif
if ($? == 1) exit

#Prep field map
if ( ! -e fmap.nii.gz || $force == 1 ) then
   fsl_prepare_fieldmap SIEMENS pha mag_masked fmap 2.46
   set force = 1
endif
if ($? == 1) exit

#Loop through metabolites
foreach met ( eg me )
   
   #Get transformations between metabolite images and anatomical
   if ( ! -e single_echo_${met}_to_anat.mat || $force == 1 ) then
      convert_xfm -concat mag_to_anat.mat single_echo_${met}_to_mag.mat \
                  -omat single_echo_${met}_to_anat.mat
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get reverse transformation
   if ( ! -e mag_to_single_echo_${met}.mat || $force == 1 ) then
      convert_xfm -omat mag_to_single_echo_${met}.mat \
                  -inverse single_echo_${met}_to_mag.mat
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get shift map
   if ( ! -e single_echo_${met}_shift.nii.gz || $force == 1 ) then
   
      #Resample field map
      flirt -in fmap -ref single_echo_${met} -applyxfm \
            -init mag_to_single_echo_${met}.mat -o fmap_on_single_echo_${met}
            
      #Scale to use for c13
      fslmaths fmap_on_single_echo_${met} -div 3.9761 fmap_on_single_echo_${met}
      
      #Convert to shift map
      if ( $met == "eg" ) then
         set dwell = 0.00082
      else
         set dwell = 0.00066
      endif
      fugue --loadfmap=fmap_on_single_echo_${met} --dwell=$dwell \
            --saveshift=single_echo_${met}_shift
            
      set force = 1
           
   endif
   if ($? == 1) exit
   
   #Apply shift map
   if ( ! -e single_echo_${met}_ud_cmplx.nii.gz || $force == 1 ) then
      python3 ../py/shift_mod.py single_echo_${met}_cmplx.nii.gz \
                                 single_echo_${met}_shift.nii.gz \
                                 single_echo_${met}_ud_cmplx
      set force = 1
   endif
   if ($? == 1) exit
   
   #Get miage after shifting
   if ( ! -e single_echo_${met}_ud_mag.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_to_plr.py single_echo_${met}_ud_cmplx.nii.gz \
                                    single_echo_${met}_ud
      set force = 1
   endif
   if ($? == 1) exit
  
   #Resample epi to anatomical
   if ( ! -e single_echo_${met}_ud_mag_on_anat.nii.gz || $force == 1 ) then
      applywarp -i  single_echo_${met}_ud_mag.nii.gz -r anat \
                --postmat=single_echo_${met}_to_anat.mat --interp=spline \
                -o single_echo_${met}_ud_mag_on_anat
      set force = 1
   endif
   if ($? == 1) exit
   
end

#Run python script to make figure
if ( ! -e fig_2.tiff || $force == 1 ) then
   python3 fig_2.py
endif
if ($? == 1) exit

