FROM python:3.5
WORKDIR ./Alpha_Domino
COPY . .
RUN pip install --upgrade setuptools
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80
ENV NAME Alphe_Domino