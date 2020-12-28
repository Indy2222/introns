#!/usr/bin/env bash

# This Bash script uses `gcloud` command from Google Cloud SDK. Visit this page
# for installation instructions:
# https://cloud.google.com/sdk/docs/quickstart-linux
#
# Note that you must have access to the data and be logged in, see
# `gcloud auth --help`

set -e

DATA_TMP_DIR=`mktemp -p /var/tmp -d`
DATA_URL='gs://thesis.mgn.cz/data'
TARGET_DIR='./data'

echo "Going to download files from ${DATA_URL} to ${DATA_TMP_DIR}…"
gsutil cp -r "${DATA_URL}/*" ${DATA_TMP_DIR}/

echo "Going to extract data…"
unzip ${DATA_TMP_DIR}/INTRON_PREDICTION.zip -d ${TARGET_DIR} &> /dev/null
mv ${DATA_TMP_DIR}/*.fa.gz ${TARGET_DIR}
gzip -d ${TARGET_DIR}/**/*.gz

echo "Going to remove temporary data…"
rm -r ${DATA_TMP_DIR}
