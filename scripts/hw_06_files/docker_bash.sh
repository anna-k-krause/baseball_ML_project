#!/bin/bash

# initial sleep to make sure mariadb container is ready
sleep 10

set -e
sleep 10
echo "Bash Script Started Successfully"

# Wait for MariaDB to start
#until mysqladmin ping -h mariadb -u root -proot -P3306 --silent; do
#    echo "Waiting for MariaDB to start..."
#    sleep 1
#done

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
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/hw_06_prep.sql
echo "SQL Features Created Successfully"

# Export final table to txt file
# source: https://www.cyberciti.biz/faq/run-sql-query-directly-on-the-command-line/
mysql -h mariadb -u root -proot -P3306 -D baseball -e "SELECT * FROM baseball.AAA_final;" > /features_table.txt
echo "SQl Select ran successfully"
# Sean said we could stop with just the final output tables from hw_05

# extra notes:
# 1. copy the sql file in container -- add in sleep for 10 seconds
# 2. connect to mariadb, connect baseballdb to mariadb container -check if already exists (makes empty db)
# 3. send baseball.sql file to to mariadb to empty db
# 4. execute sql file
# 5. execute python file (should be in cmd command)