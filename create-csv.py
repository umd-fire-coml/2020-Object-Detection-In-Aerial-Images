# %%
import pandas as pd
import re
import os

# %%
annotations_location = 'test' # GIVE ME THE PATH thanks
csv = pd.DataFrame(columns=['img_source', 'gsd', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category', 'is_difficult'])
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
                    csv = csv.append({'img_source': source, 
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
            print("Please delete previous annotations before running this again!")
        else:
            print("This file bad: ", annotation_path)
        good = False

if good:
    save_location = os.path.join(annotations_location, 'annotations.csv')
    csv.to_csv(save_location, index=False, index_label=False)
    print("Done with ", annotations_location)
else:
    print("Not good, fix stuff before csv, thanks.")