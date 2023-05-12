#!/bin/bash

# initial sleep to make sure mariadb container is ready
sleep 10

set -e
sleep 10
echo "Bash Script Started Successfully"

# Create baseball db in mariadb container
# source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/5/0/0
mysql -h mariadb -u root -proot -P3306 -e "CREATE Database baseball;"
echo "Baseball Database Created Successfully"
sleep 10

# Copy over database from baseball.sql file
# set timeout limit, so the code won't timeout
# source: https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/5/0/0
mysql -h mariadb -u root -proot -P3306 -e "SET GLOBAL net_read_timeout=6000;"
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/baseball.sql
echo "Database Copied Successfully"
sleep 10

# Run Homework SQL file
# source: https://stackoverflow.com/questions/7616520/how-to-execute-a-sql-script-from-bash
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/final_prep.sql
echo "SQL Features Created Successfully"

# Export final table to txt file
# source: https://www.cyberciti.biz/faq/run-sql-query-directly-on-the-command-line/
mysql -h mariadb -u root -proot -P3306 -D baseball -e "SELECT * FROM baseball.AAA_final;" > /features_table.txt
echo "SQl Select ran successfully"

# Run python file
python3 final_main.py > /results_table.txt
echo "Python code ran successfully"

# extra notes:
# sorry if this doesn't work, I tried my best