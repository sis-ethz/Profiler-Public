install:
	pip install -r requirements.txt
	python -m ipykernel install --user --name fd37 --display-name "Python (fd37)"
	pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl
	conda install -c conda-forge suitesparse
	pip install scikits.sparse
	git clone https://github.com/scikit-sparse/scikit-sparse.git
	python scikit-sparse/setup.py install
