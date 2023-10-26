#!/bin/bash

if [[ "$1" == "--overwrite" ]]; then
    echo "Deleting ./database/chroma.sqlite3..."
    rm ./database/chroma.sqlite3
    echo "File deleted."
fi

echo "Running database/db_create.py..."
python database/db_create.py
echo "Script execution completed."
