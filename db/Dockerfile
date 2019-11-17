FROM postgres

MAINTAINER Mikhail Smirnov

# These files will be executed by postgre, so we don't have to run them ourselves
COPY ./init /docker-entrypoint-initdb.d

# Set persmissions and run
# RUN chmod +x /var/lib/pgsql/data/restore.sh

# open port 5432
EXPOSE 5432
