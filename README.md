# CV Detection

This is the NYU Robomaster's code for the Robomasters Robotics Competition.

## Table of Contents

- [Pipeline Documentation](./Documentation/Pipeline/Pipeline.md)
- [Resources to get started](./Documentation/Resources/Resources.md)

### Create docker image

#### Only build once

```shell
cd Docker
docker/build.sh
```

#### Run docker image

```shell
docker/run.sh --volume ./Docker/:/jetson_interface -c nyuultraviolet.cv-dockerfile:xavier
```

#### list current docker images

```shell
sudo docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
```

#### export docker image

```shell
sudo docker save -o CV.tar <IMAGE_ID>
```

#### import docker image

```shell
sudo docker load -i CV.tar
```

### Update yml files

Incase the current yml files do not setup the conda environment, create a issue and commit a new yml file that works for that OS. Specify what was breaking in the PR.

#### Linux or Mac

```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
```

#### Windows

```shell
conda env export --no-builds | findstr -v "prefix" > environment.yml
```
