echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8" # change py version as per your need
conda create --prefix ./mlexhaust python=3.9 -y
echo [$(date)]: "activate mlexhaust"
source activate ./mlexhaust
echo [$(date)]: "installing the requirements" 
pip install -r requirements.txt
echo [$(date)]: "END" 
