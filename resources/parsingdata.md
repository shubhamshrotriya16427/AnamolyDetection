## Steps to parse data downloaded from RIPE

### Install mrtparse
1. Install mrtparse library.
```
>>> pip install mrtparse
```
2. Clone the repo https://github.com/t2mune/mrtparse/tree/master locally. In examples folder, it has files to parse the data that we'll eventually get.

### Unzipping and parsing data

1. Download the .gz files from https://data.ris.ripe.net/rrc04/
2. Unzip these files
    ```
    >>> gunzip bview.20010601.0043.gz
    ```
3. The above will produce a file bview.20010601.0043 in the same directory.
4. Use the `mrt2exabgp.py` file from the cloned repo to parse the unzipped file.
```
    >>> python mrt2bgpdump.py -M -O bview.20010601.0043_parsed.txt ../../bview.20010601.0043  
```

