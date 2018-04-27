FROM python:3-onbuild

RUN apt-get update
RUN apt-get install -q -y net-tools
RUN pip install requests
RUN pip install scikit_learn
RUN pip install scipy
RUN pip install numpy
RUN pip install pandas





COPY . /app 

WORKDIR /app

EXPOSE 8080

CMD ["make", "run"]
