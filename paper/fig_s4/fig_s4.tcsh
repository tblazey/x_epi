#!/bin/tcsh -f

#Usage
if ( $#argv == 1 ) then
    set force = $1
else
    set force = 0
endif

set out = slice

#Run recon
if ( ! -e ${out}_eg_cmplx.nii.gz || $force == 1 ) then
   x_epi_recon meas_MID02185_FID226126_pulseq.dat seq_for_recon.json ${out} \
               -n_avg 128 -anat mag.nii.gz -complex
   set force = 1
endif

#Use anatomical from figure 2
ln -sf ../fig_2/anat.nii.gz anat.nii.gz
ln -sf ../fig_2/anat_mask.nii.gz anat_mask.nii.gz
                         
#Register mag to anatomical
if ( ! -e mag_to_anat.mat || $force == 1 ) then
   flirt -in mag -ref anat -dof 6 -omat mag_to_anat.mat \
         -o mag_on_anat
   convert_xfm -omat anat_to_mag.mat -inverse mag_to_anat.mat
   set force = 1
endif

#Get magnitude mask
if ( ! -e mag_mask.nii.gz || $force == 1 ) then
   flirt -in anat_mask -ref mag -applyxfm -init anat_to_mag.mat -o mag_mask
   fslmaths mag_mask -thr 0.5 -bin mag_mask
   fslmaths mag -mas mag_mask mag_masked
   set force = 1
endif

#Prep field map
if ( ! -e fmap.nii.gz || $force == 1 ) then
   fsl_prepare_fieldmap SIEMENS pha mag_masked fmap 2.46
   set force = 1
endif

#Loop through metabolites
foreach met ( eg me )
   
   #Get transformations between metabolite images and anatomical
   if ( ! -e ${out}_${met}_to_anat.mat || $force == 1 ) then
      convert_xfm -concat mag_to_anat.mat ${out}_${met}_to_mag.mat \
                  -omat ${out}_${met}_to_anat.mat
      set force = 1
   endif
   
   #Get reverse transformation
   if ( ! -e mag_to_${out}_${met}.mat || $force == 1 ) then
      convert_xfm -omat mag_to_${out}_${met}.mat \
                  -inverse ${out}_${met}_to_mag.mat
      set force = 1
   endif
   
   #Get shift map
   if ( ! -e ${out}_${met}_shift.nii.gz || $force == 1 ) then
   
      #Resample field map
      flirt -in fmap -ref ${out}_${met} -applyxfm \
            -init mag_to_${out}_${met}.mat -o fmap_on_${out}_${met}
            
      #Scale to use for c13
      fslmaths fmap_on_${out}_${met} -div 3.9761 fmap_on_${out}_${met}
      
      #Convert to shift map
      if ( $met == "eg" ) then
         set dwell = 0.00082
      else
         set dwell = 0.00066
      endif
      fugue --loadfmap=fmap_on_${out}_${met} --dwell=$dwell \
            --saveshift=${out}_${met}_shift
            
      set force = 1
           
   endif
   
   #Apply shift map
   if ( ! -e ${out}_${met}_ud_cmplx.nii.gz || $force == 1 ) then
      python3 ../py/shift_mod.py ${out}_${met}_cmplx.nii.gz \
                                 ${out}_${met}_shift.nii.gz \
                                 ${out}_${met}_ud_cmplx
      set force = 1
   endif
   
   #Get mean after shifting
   if ( ! -e ${out}_${met}_ud_mag.nii.gz || $force == 1 ) then
      python3 ../py/cmplx_to_plr.py ${out}_${met}_ud_cmplx.nii.gz \
                                    ${out}_${met}_ud
      set force = 1
   endif

   #Resample epi to anatomical
   if ( ! -e ${out}_${met}_ud_mag_on_anat.nii.gz || $force == 1 ) then
      applywarp -i  ${out}_${met}_ud_mag.nii.gz -r anat \
                --postmat=${out}_${met}_to_anat.mat --interp=spline \
                -o ${out}_${met}_ud_mag_on_anat
      set force = 1
   endif
   
end

#Run python script to make figure
if ( ! -e fig_s4.tiff || $force == 1 ) then
   python3 fig_s4.py
endif

