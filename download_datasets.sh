#!/bin/sh
echo 1. Downloading the datasets in a temp folder
TMP="$(mktemp)"
echo "Downloading in $TMP"
wget https://zenodo.org/record/3484797/files/20160809.zip -P $TMP
wget https://zenodo.org/record/3484797/files/20160810.zip -P $TMP

echo 2. Checking the datasets
echo "5643002a8893ff5c3985bb0d2132afd9  20160809.zip\nf054bd7b86d47b97821f5c14e0eecd77  20160810.zip" > $TMP/checksum.md5
md5sum -c $TMP/checksum.md5
unzip -tq $TMP/20160809.zip
unzip -tq $TMP/20160810.zip

echo 3. Decompressing the datasets
mkdir -p datasets/{training,testing,validation}
unzip $TMP/20160809.zip -d datasets/training
unzip $TMP/20160810.zip -d datasets/testing

echo 4. Moving files to create test set
mv datasets/testing/20160810-002-xy6* datasets/validation

echo 5. Deleting temp directory
rm -r $TMP
