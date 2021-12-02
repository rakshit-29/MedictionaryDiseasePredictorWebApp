FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt


RUN pip3 install -r requirements.txt

EXPOSE 2929

COPY . /app


CMD streamlit run --server.port 2929 --server.enableCORS false app.py
