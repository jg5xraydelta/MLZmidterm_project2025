docker build -t heart-predict .

docker run -it --rm -p 9696:9696 heart-predict