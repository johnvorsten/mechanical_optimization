## About

## Testing, linting, automation
from 'src' directory: `python -m unittest discover -p "test_*.py"`
python -m unittest tests/test_data_load.py
coverage run -m unittest tests/test_regression.py
pylint -r n chiller_optimization/regression.py