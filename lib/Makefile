test:
	jupyter nbconvert --to python lib.ipynb
	mv lib.py test_lib.py
	pytest -s --doctest-modules test_lib.py
	
