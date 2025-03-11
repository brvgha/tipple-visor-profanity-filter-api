sudo yum update -y
mkdir ./api
sudo yum install git -y
git clone https://github.com/brvgha/tipple-visor-profanity-image-filter-api
cd ./tipple-visor-profanity-image-filter-api
aws s3 cp s3://keras-model/profanify_filter_model.keras ./profanify_filter_model.keras
sudo yum install -y python3
sudo yum install -y python3-pip
python3 -m venv myenv
source venv311/bin/activate
pip install tensorflow fastapi uvicorn pillow python-multipart pydantic
uvicorn ml_api:app --host 0.0.0.0 --port 8000