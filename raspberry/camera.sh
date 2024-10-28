#!/bin/bash
libcamera-vid -t 0 --width 640 --height 480 --framerate 30 -o - | cvlc -vvv stream:///dev/stdin --sout '#standard{access=http,mux=ts,dst=:8082}' :demux=h264