sudo yum update -y
sudo yum install -y python3
sudo yum install -y python3-pip
python3 -3.11 -m venv venv311 
source venv311/bin/activate
pip install tensorflow fastapi uvicorn pillow
uvicorn ml_api:app --host 0.0.0.0 --port 8000