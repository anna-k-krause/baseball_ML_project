
FROM python:3.9.6
#FROM ubuntu:latest
# may have problems with python container -maybe make umbuntu

# I had to run these first? Not sure if you'll need to
# docker-compose up --build --force-recreate --no-deps -d mariadb
# docker-compose up --build --force-recreate --no-deps -d db_cli

ENV APP_HOME /app
WORKDIR $APP_HOME

# Get necessary system packages
# source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture13.html#/6/7/9
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY baseball.sql .
COPY hw_06_prep.sql .

# Run bash script
# source: https://stackoverflow.com/questions/60485743/how-to-use-docker-entrypoint-
# source cont: with-shell-script-file-combine-parameter
COPY docker_bash.sh /
RUN chmod +x /docker_bash.sh
ENTRYPOINT ["/docker_bash.sh"]
