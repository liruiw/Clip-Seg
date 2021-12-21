# Download the data
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1gmx7uT7VFFoW_X4q9pI0J3uve-dA4ASs?' -O data.zip
echo "Data downloaded. Starting to unzip"
unzip  data.zip  -d output
rm data.zip
