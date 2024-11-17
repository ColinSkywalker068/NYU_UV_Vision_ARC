**Warning: The dockerfiles here are deprecated. Go to uv-services/Docker to find the most up-to-date dockerfiles**

# Docker Basics

Search confluence and docker documentation, the commands are usually the same.

## Building an image

```
cd <folder-containing-dockerfiles>
sudo docker -t <image tag> -f <name-of-dockerfile> .
```

## Test image
For quick tests:
```
sudo docker run <image tag / image hash> -it 
```

If you want GUI, CUDA and camera support with your docker container, run the following from the base of this repository
```
cd docker
sudo docker/run.sh -c <image tag / image hash>
```

## Publish image
Login to nyuultraviolet account using CV Credentials 
```
sudo docker login
sudo docker tag <image tag / image hash> <nyuultraviolet/project-name:tag>
sudo docker push <nyuultraviolet/project-name:tag>
```

## CV base image
We use https://github.com/dusty-nv/jetson-containers/tree/master to build our base containers from scratch on the AGX orin. The realsense containers do not work as expected hence we have our own `Dockerfile.realsense`.

For questions on how to use jetson-containers contact `dusty-nv` using github issues, he's pretty responsive and look at already open / closed issues before making a new issue. Most problems are usually documented before by someone else.