test:
	jupyter nbconvert --to python lib.ipynb
	mv lib.py test_lib.py
	pytest --doctest-modules test_lib.py
	
