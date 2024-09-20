import re
import chardet
import csv

samples = []

# Detect the encoding of the file
with open('Source.txt', 'rb') as rawfile:
    rawdata = rawfile.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Function to clean up a line
def clean_line(line):
    # Strip unwanted characters and stop at '/'
    new_string = ""
    for char in line:
        if char == "/":
            break
        new_string += char
    return new_string.strip()

# Read the file using the detected encoding
with open('Source.txt', encoding=encoding) as myfile:
    lines = myfile.readlines()

# Clean the lines and remove unwanted characters
cleaned_lines = [clean_line(line) for line in lines]

# Create pairs of consecutive lines for the dataset
new_list = []
i = 0
count = 1
while i < len(cleaned_lines) - 1:
    # Create a pair of lines, the poem and its next line
    temp1 = cleaned_lines[i] + "," + cleaned_lines[i + 1]
    temp = f"{count},{temp1}"
    new_list.append([str(count), temp1])  # Store as a list to avoid character separation
    count += 1
    i += 2  # Jump two lines each iteration

# Write the data to CSV with UTF-8 BOM encoding
with open("dataset.csv", mode="w", encoding="utf-8-sig", newline="") as file:  
    writer = csv.writer(file)
    writer.writerow(["ID", "X"])  # adding id and feature tag
    writer.writerows(new_list)

print("Data has been successfully written to data.csv with UTF-8 BOM")
