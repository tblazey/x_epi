Command Line
------------

The command line program **x_epi_cmd.py** can be used to generate an Pulseq 'seq' file.
The program has several options that can be divided into either
:doc:`general parameters <gen>` that apply to all metabolites and 
:doc:`metabolite specific <met>` parameters. Multiple parameters can be specified using
multiple -met flags.

For example, the following code block creates a sequence that uses a symmetric 3D readout
to acquire a 320 cm\ :sup:`3` FOV. Two separate metabolites are acquired. The first
metabolite, 'pyr', uses a grid size of 16x16x16 and a flip angle of 10 degrees. The 
second metabolite, 'lac', has a larger flip angle (45 degrees) and a smaller grid size
(12x12x12).

.. code-block:: bash

   x_epi_cmd.py -out example -fov 320 320 320 -symm_ro -acq_3d -met -name 'pyr' \
                -size 16 16 16 -flip 10 -met -name 'lac' -size 12 12 12 -flip 45


In addition to the Pulseq '.seq' file, the script also outputs a JSON parameters file
and a Numpy data file containing the k-space coordinates.

.. toctree::
   :maxdepth: 1
   
   gen
   met
