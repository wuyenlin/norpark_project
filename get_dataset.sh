#!/bin/sh

which unzip > /dev/null && which wget > /dev/null || { echo "Please install wget and unzip" && exit 1; }
# 1. CNRPark-EXT dataset
wget "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "dataset.zip"
unzip dataset.zip
rm dataset.zip

# 2. PKLot dataset
wget "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz" -O "PKLot.tar.gz"
tar -xvf PKLot.tar.gz
rm PKLot.tar.gz