#!/bin/sh
echo Setting up ssh with ngrok
sudo service ssh start
ngrok tcp 22
