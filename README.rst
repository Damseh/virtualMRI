=========
VirtualMRI
=========

A Python package to simulate diffusion MRI response from vascular graphs.

Papers
------
@article {Damseh2020.08.19.257741,
	author = {Damseh, Rafat and Lu, Yuankang and Lu, Xuecong and Zhang, Cong and Marchand, Paul J. and Corbin, Denis and Pouliot, Philippe 		and Cheriet, Farida and Lesage, Frederic},
	title = {A Simulation Study Investigating Potential Diffusion-based MRI Signatures of Microstrokes},
	elocation-id = {2020.08.19.257741},
	year = {2020},
	doi = {10.1101/2020.08.19.257741},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/08/20/2020.08.19.257741},
	eprint = {https://www.biorxiv.org/content/early/2020/08/20/2020.08.19.257741.full.pdf},
	journal = {bioRxiv}}

To install
----------

``conda create -n ENV_NAME python=3.7.5`` 

``conda activate ENV_NAME``

``conda install -c conda-forge mayavi=4.6.2``

``conda install -c conda-forge spyder=4.0.1 matplotlib=3.1.1 scipy=1.3.1 networkx=2.2 scikit-learn pandas tqdm``

``pip install xlrd==1.2.0``

``git clone https://github.com/Damseh/VascularGraph.git``

``git clone https://github.com/flesage/virtualMRI.git``

``cd VascularGraph``

``python setup.py install``

``cd ../virtualMRI``

``python setup.py install``

To test
-------

``python -i test_oct.py``

* Free software: MIT license

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
