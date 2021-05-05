FROM python:3

ENV DEBIAN_FRONTEND=noninteractive

COPY . /app
WORKDIR /app

#RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

#ENTRYPOINT ["python","/app/main.py"]
ENTRYPOINT ["bash","/app/scripts/run_task_multichoice_haihua_predict.sh"]