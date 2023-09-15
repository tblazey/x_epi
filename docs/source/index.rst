X_EPI Documentation
==================================

.. figure:: static/x_epi_logo.png
   :width: 300
   :align: center
   
   Logo designed by Alyssa Weisenstein.

**x_epi** is a Python library for creating echo-planar imaging (EPI) sequences for 
X-nuclei. It uses the `Pulseq framework <https://pulseq.github.io>`_ implemented in 
`PyPulseq <https://github.com/imr-framework/pypulseq/>`_.

Users can create custom EPI sequences using :doc:`several different methods <seq_doc/index>`.
Multiple features are available including:

#. Spectral Spatial Radio-frequency (SSRF) excitation
#. Symmetric or flyback readouts with an arbitrary number of echoes
#. 2D or 3D imaging
#. Partial Fourier in either phase encoding direction
#. Optional ramp sampling

Although this package was designed for imaging hyperpolarized :sup:`13`\C agents, it can
be used for several different nuclei. Please feel free to 
`ask <https://github.com/tblazey/x_epi/discussions>`_ if you have any questions!

.. toctree::
   :maxdepth: 1
   :hidden:
   
   install
   seq_doc/index
   recon_doc/index
   tutorial
   

   
