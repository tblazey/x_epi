<div align="center">
      <img src="docs/source/static/x_epi_logo.png" alt="drawing" width="250"/>
</div>
<div align="center">
      <i>Logo designed by Alyssa Weisenstein</i></p>
</div>

&nbsp;

<div align="center">   

[![Documentation Status](https://readthedocs.org/projects/x-epi/badge/?version=latest)](https://x-epi.readthedocs.io/en/latest/?badge=latest)
[![Pylint](https://github.com/tblazey/x_epi/actions/workflows/pylint.yml/badge.svg)](https://github.com/tblazey/x_epi/actions/workflows/pylint.yml)
[![codecov](https://codecov.io/gh/tblazey/x_epi/graph/badge.svg?token=GS8QK3LG16)](https://codecov.io/gh/tblazey/x_epi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Static Badge](https://img.shields.io/badge/10.1002%2Fmrm.30090-doi?label=doi&labelColor=828282&color=0f52ba&cacheSeconds=https%3A%2F%2Fdoi.org%2F10.1002%2Fmrm.30090)](https://doi.org/10.1002/mrm.30090)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**x_epi** is a Python package for creating EPI sequences for X-nucleus imaging. It was designed for hyperpolarized <sup>13</sup>C imaging, but can be used for <sup>1</sup>H and other NMR sensitive nuclei. It uses the [PyPulseq](https://github.com/imr-framework/pypulseq) package to create vendor neutral sequences in the [Pulseq](https://pulseq.github.io) format. 

Users can create a custom EPI sequence using python functions, a command line program, or a simple graphical user interface. 

A paper describing the sequence/program has been [published](https://doi.org/10.1002/mrm.30090) in Magnetic Resonance in Medicine.

## Installation

If you have Python 3.10, you can install the package using pip:

```
pip install git+https://github.com/tblazey/x_epi.git
```

Please see the [documentation](https://x-epi.readthedocs.io/en/latest/install.html) for more details on installation.

## Quick start

Once installed, you can type `x_epi_gui` in a command prompt to launch the user interface. You will then be prompted to either load a custom JSON configuration file or use the default. If you don't have a JSON configuration, simply click 'Use Default' to get started.

You can also create a sequence using `x_epi_cmd`:

```
x_epi_cmd -out example -fov 320 320 320 -symm_ro -acq_3d -met -name 'pyr' \
          -size 16 16 16 -flip 10 -met -name 'lac' -size 12 12 12 -flip 45
```

The command above creates a 3D EPI sequence with two different metabolites. For more
information about creating sequences, see the [documentation](https://x-epi.readthedocs.io/en/latest/seq_doc/index.html)