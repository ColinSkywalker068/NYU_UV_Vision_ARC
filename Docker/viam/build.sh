BASE_IMAGE=python
	
# build the container
sudo docker build -t $TAG -f Dockerfile \
		.
