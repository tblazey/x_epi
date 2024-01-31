#!/bin/tcsh -f
set force = 0
#Loop through figure directories
foreach fig ( `ls -d fig_*` )

   #Move into directory
   pushd $fig
   
   #Run tcsh if available, otherwise just make figure
   if ( ! -e ${fig}.tiff || $force == 1 ) then
      if ( -e sim_pulse.jl && (! -e pulse_sim.npy || $force == 1) ) then
         julia sim_pulse.jl
      endif
      if ( -e ${fig}.tcsh ) then
         ./${fig}.tcsh
      else
         python3 ${fig}.py
      endif
   endif
   
   popd

end
