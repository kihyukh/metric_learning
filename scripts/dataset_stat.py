import argparse
import os

parser = argparse.ArgumentParser(description='Print dataset stats')
parser.add_argument('--dir', help='directory containing data')
args = parser.parse_args()

image_count = 0
class_count = 0
for subdir, dirs, files in os.walk(args.dir):
    for file in files:
        image_count += 1
    for dir in dirs:
        class_count += 1

print('Image count: {}'.format(image_count))
print('Class count: {}'.format(class_count))
