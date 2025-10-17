# Windows
docker run -it --rm --privileged --network host ^
    --volume %~dp0..\:/home/nyu/CV_Detection ^
    --env DISPLAY=host.docker.internal:0.0 ^
    nyuultraviolet/cv-dockerfile:nav-amd64
