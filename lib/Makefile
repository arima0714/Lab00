test:
	jupyter nbconvert --to python lib.ipynb
	mv lib.py lab_lib.py
	pytest -s --doctest-modules lab_lib.py
	
