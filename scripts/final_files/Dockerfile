
FROM python:3.9.6

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture13.html#/6/7/9
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     mariadb-client \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Get necessary system packages
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY baseball.sql .
COPY final_prep.sql .
COPY final_main.py .

# Run bash script
# source: https://stackoverflow.com/questions/60485743/how-to-use-docker-entrypoint-
# source cont: with-shell-script-file-combine-parameter
COPY docker_bash.sh /
RUN chmod +x /docker_bash.sh
ENTRYPOINT ["/docker_bash.sh"]
