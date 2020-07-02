#!/bin/sh

which unzip > /dev/null && which wget > /dev/null && which tar > /dev/null || { echo "Please install wget, unzip, and tar" && exit 1; }

# 1. CNRPark dataset
wget "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip" -O "CNRPark.zip"
unzip CNRPark.zip && rm CNRPark.zip

# 2. CNRPark-EXT dataset
wget "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "CNRPark_EXT.zip"
unzip CNRPark_EXT.zip && rm CNRPark_EXT.zip

# 3. PKLot dataset
wget "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz" -O "PKLot.tar.gz"
tar -xvf PKLot.tar.gz && rm PKLot.tar.gz
