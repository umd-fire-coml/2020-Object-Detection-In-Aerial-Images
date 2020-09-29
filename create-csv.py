# %%
import pandas as pd
import re
import os
import csv

# %%
annotations_location = 'test' # GIVE ME THE PATH thanks
csv_final = pd.DataFrame(columns=['pic', 'img_source', 'gsd', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category', 'is_difficult'])
good = True

for annotation_names in os.listdir(annotations_location):
    annotation_path = os.path.join(annotations_location, annotation_names)
    try:
        with open(annotation_path) as current:
            first_line = True
            source = None
            gsd = None
            second_line = False
            for line in current:
                if first_line:
                    source = re.search("^imagesource:(.*)$", line).group(1)
                    second_line = True
                    first_line = False
                elif second_line:
                    gsd = re.search("^gsd:(.*)$", line).group(1) # a little hardcoded but it's still good!
                    second_line = False
                else:
                    # Bad regex ahead (idk how to capture multiple repetitions)
                    to_store = re.search("^(\d*\.\d) (\d*\.\d) (\d*\.\d) (\d*\.\d) (\d*\.\d) (\d*\.\d) (\d*\.\d) (\d*\.\d) (.*) (\d)$", line)
                    csv_final = csv_final.append({'pic': annotation_names,
                                'img_source': source, 
                                'gsd': gsd, 
                                'x1': to_store.group(1), 
                                'y1': to_store.group(2), 
                                'x2': to_store.group(3), 
                                'y2': to_store.group(4), 
                                'x3': to_store.group(5), 
                                'y3': to_store.group(6), 
                                'x4': to_store.group(7), 
                                'y4': to_store.group(8), 
                                'category': to_store.group(9), 
                                'is_difficult': to_store.group(10)}, 
                                ignore_index=True)
    except:
        if annotation_path == os.path.join(annotations_location, 'annotations.csv'):
            print("Please delete previous annotations file before running this again!")
        else:
            print("This file bad: ", annotation_path)
        good = False

if good:
    save_location = os.path.join(annotations_location, 'annotations.csv')
    csv_final.to_csv(save_location, index=False, index_label=False)
    print("Done with ", annotations_location)
else:
    print("Not good, fix stuff before csv, thanks.")

# %%
# Hash filename to a 2d array containing gsd, then all the bounding box coords, then category.
def parse_csv(csv_path, data_path):
    to_return = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader) # Skip header, cuz who needs it
        for line in reader:
            curr = [line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11]]        
            if line[0].strip('.txt') in to_return:
                to_return[line[0].strip('.txt')].append(curr)
            else:
                to_return[line[0].strip('.txt')] = [curr]
    return to_return

# %%
parse_csv('./test/annotations.csv', None)
# %%
