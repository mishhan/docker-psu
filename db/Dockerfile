FROM postgres

MAINTAINER Mikhail Smirnov

# These files will be executed by postgre, so we don't have to run them ourselves
COPY ./init /docker-entrypoint-initdb.d

EXPOSE 5432