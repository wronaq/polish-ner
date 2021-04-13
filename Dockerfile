FROM ubuntu:latest
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install python3-pip -y \
    && pip3 install --upgrade pip \
    && apt-get autoclean \
    && apt-get autoremove -y \
    && mkdir -p /usr/src/app
WORKDIR /usr/src/
COPY app ./app
COPY weights ./weights
COPY tasks ./tasks
COPY polish_ner ./polish_ner
COPY data ./data
COPY requirements.txt ./
EXPOSE 5000
RUN pip3 --no-cache-dir install -r requirements.txt
ENV PYTHONPATH=/usr/src/
ENTRYPOINT ["python3"]
CMD ["app/app.py"]
