
FROM python:3.9.6
#FROM centos:latest
#FROM ubuntu:latest
# may have problems with python container -maybe make umbuntu

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
# COPY requirements.txt .
#RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY baseball.sql baseball.sql
COPY hw_06_prep.sql hw_06_prep.sql
COPY hw_06_main.py hw_06_main.py

# Create an unprivileged user
COPY docker_bash.sh /
RUN chmod +x /docker_bash.sh
#ENTRYPOINT echo "entrypoint test"
ENTRYPOINT ["/docker_bash.sh"]

# RUN ["cp ./baseball.sql mariadb:/", "sleep 10"]
# RUN ["mysql -u root -p -e CREATE DATABASE baseball"]
# RUN ["mysql -u root -p baseball < baseball.sql"]
# 1. copy the sql file in container -- add in sleep for 10 seconds
# 2. connect to mariadb, connect baseballdb to mariadb container -check if already exists (makes empty db)
# 3. send baseball.sql file to to mariadb to empty db
# 4. execute sql file
# 5. execute python file (should be in cmd command)


# Run app
# CMD ["mysql -u root -p baseball < baseball.sql"]


# source : https://medium.com/cranecloud/docker-django-and-gunicorn-61275f4ea14c

# execute mariadb container -> put baseball.sql in it?? -> run sql file in mariadb container -> run python file (needs to connect to sql)

# what version of baseball to do put in my script? tarball or unzipped?