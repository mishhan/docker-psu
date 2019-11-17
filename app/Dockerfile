FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
# takes a lot of time to install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./app.py" ]
