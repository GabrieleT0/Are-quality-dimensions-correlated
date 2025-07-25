# Create and activate a Python virtual environment
python3 -m venv venv
source ./venv/bin/activate
# Install all the dependencies
pip install -r requirements.txt
# Calculate the correlation
cd src
python3 main.py