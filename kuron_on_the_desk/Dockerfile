FROM python:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake clang libssl-dev vim

COPY ./requirements.txt /app/requirements.txt

# Install system dependencies
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}

CMD [ "python", "./server/kuron.py" ]