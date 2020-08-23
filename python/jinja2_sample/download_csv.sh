# Download diabetes dataset.
#
# Usage: bash download_csv.sh


readonly URL=https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
readonly FILEPATH=data/raw/diabetes.csv

wget $URL -O $FILEPATH
