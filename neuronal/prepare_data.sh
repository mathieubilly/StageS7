#!/bin/bash

if [ $1 = "help" ]
then
    echo "Usage: ./prepare_data input"
else 

sed -i 's/\t/,/g' $1
sed -i 's/  /,/g' $1
sed -i 's/ /,/g' $1

fi



