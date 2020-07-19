FROM python:3.7

WORKDIR /app

COPY text_renderer /app/text_renderer
COPY setup.py /app/setup.py
COPY main.py /app/main.py
COPY tools /app/tools
COPY example_data /app/example_data
COPY docker /app/docker

RUN pip3 install -r docker/requirements.txt && \
    python3 setup.py develop

ENV TERM xterm-256color
CMD sh docker/run.sh
