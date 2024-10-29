@echo off
echo Setting up the environment...

if not exist "venv" (
    echo Creating a virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Running train_fabric.py...
python train_fabric.py

pause
