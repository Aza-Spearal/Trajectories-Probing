#!/bin/bash

fichier="temp.txt" #in the beginning 'temp.txt' should be at 0
indice=$(cat "$fichier")


while true; do
    echo "$indice" | poetry run python3 saver_core.py

    out=$(cat "$fichier")
    echo $out

    if [[ "$out" == "0" ]]; then
        break
    fi

    indice=$out

done
