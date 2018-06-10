#!/usr/bin/bash

# put your own user id here
USER=mdrissi 

# first parameter is the command to run
# additional parameters will be tacked on to the 
# command so that you can pass parameters to your program
CMD=$1
echo running: $CMD $2 $3 $4 $5

# start the docker run command
# (--devices) load devices
# (-v) load libraries
# (-v) mount user volume
# specify docker image
# specify command and any parameters (up to 4)
#gw000/keras:2.1.3-py2-tf-gpu
nvidia-docker run -it --rm \
	$(ls /dev/nvidia* | xargs -I{} echo '--device={}') \
	$(ls /usr/lib/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') \
	-v /home/$USER:/home/$USER \
	ufoym/deepo:latest \
	$CMD $2 $3 $4 $5 
