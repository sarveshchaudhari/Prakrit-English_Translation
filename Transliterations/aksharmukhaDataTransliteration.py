import csv

# Open the input text file with utf-8 encoding
file = open("D:/VIIT/Ty/SEM1/PR1/Prakrit-English_Translation/Transliterations/GRETIL_RAW.txt", "r", encoding='utf-8')

# Open the CSV file with utf-8 encoding
csvFile = open("data.csv", "w", encoding="utf-8")

# Initialize the CSV writer
writer = csv.writer(csvFile)

# Write each line from the text file to the CSV file
for strings in file:
    writer.writerow([strings.strip()])

# Close the files
file.close()
csvFile.close()
