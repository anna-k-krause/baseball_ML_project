#!/bin/bash
# The baseball.sql has to be in the same folder than this file

# 1. copy the sql file in container -- add in sleep for 10 seconds
# 2. connect to mariadb, connect baseballdb to mariadb container -check if already exists (makes empty db)
# 3. send baseball.sql file to to mariadb to empty db
# 4. execute sql file
# 5. execute python file (should be in cmd command)

set -e

# Wait for MariaDB to start
until mysqladmin ping -h mariadb -u root -proot -P3306 --silent; do
    echo "Waiting for MariaDB to start..."
    sleep 1
done

# service mariadb start
mysql -h mariadb -u root -proot -P3306 -e "CREATE Database baseball;"
echo "Database created successfully"
#create database from baseball.sql file
mysql -h mariadb -u root -proot -P3306 -e "SET GLOBAL net_read_timeout=600;"
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/baseball.sql
echo "Database copied successfully"

sleep 15
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/hw_06_prep.sql
echo "SQL features created"
#export final table to results.txt
mysql -h mariadb -u root -proot -P3306 -Dbaseball -e "SELECT * FROM baseball.AAA_final;" > /results.txt

echo "SQl select run"