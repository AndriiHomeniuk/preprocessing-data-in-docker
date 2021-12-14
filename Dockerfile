FROM python:3.8.8

RUN apt-get update && apt-get install dos2unix -y

COPY . .

RUN chmod +x ./start.sh
RUN pip install -r requirements.txt

CMD ["/start.sh"]
