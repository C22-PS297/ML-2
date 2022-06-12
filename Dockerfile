# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir -r /code/requirements.txt

# 
COPY ./app /code/app

RUN apt-get update
RUN pip install --upgrade pip
RUN apt-get install ffmpeg libsm6 libxext6  -y

# 
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--reload"]
