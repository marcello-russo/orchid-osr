# Script to setup a local environment for Mac with MPS support
# Run with: source setup_mac_native.sh

# 1. Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found via 'python3'. Please install it."
    return
fi

# 2. Create virtual environment
echo "Creating virtual environment 'venv_mac'..."
python3 -m venv venv_mac

# 3. Activate virtual environment
echo "Activating virtual environment..."
source venv_mac/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch with MPS support (Nightly often has better updates, but stable is fine)
# Installing stable 2.2+ which has good MPS support
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# 6. Install other requirements from the file, filtering out incompatible ones if any
# (For now assuming requirements.txt is standard)
echo "Installing other requirements..."
pip install -r requirements.txt

# 7. Create local .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env from example (empty)..."
    touch .env
fi

echo ""
echo "Setup complete!"
echo "To use this environment, run: source venv_mac/bin/activate"
echo "Then start jupyter: jupyter notebook"
