import os
import csv
import pandas as pd

# Create new csv files containing only the annotations needed for training, validation and testing
# based on the classes in mapping


def get_clean_annotation_boxes(path, mode, class_map):
    """
    Create new csv files containing only the annotations needed
    """
    df = pd.read_csv(os.path.join(path, mode, mode + '-annotations-bbox.csv'))
    new_df = df[(df['LabelName'].isin(class_map.keys()))]
    new_df.to_csv(os.path.join(path, mode, 'cleaned-annotations-bbox.csv'), index=False)


def get_clean_imageids(path, mode):
    """
    Get file with all image ids
    """
    df = pd.read_csv(os.path.join(path, mode, 'cleaned-annotations-bbox.csv'))
    df = mode + '/' + df['ImageID'].astype(str)
    df.to_csv(os.path.join(path, mode, 'images.txt'), index=False, header=False)


def get_labels(path, mode, mapping):
    """
    Get file with all labels in YOLO format
    """
    with open(os.path.join(path, mode, 'cleaned-annotations-bbox.csv')) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            c = mapping[row['LabelName']]
            width = float(row['XMax']) - float(row['XMin'])
            height = float(row['YMax']) - float(row['YMin'])
            xcenter = float(row['XMin']) + (width / 2)
            ycenter = float(row['YMin']) + (height / 2)
            id = row['ImageID']

            file = open(os.path.join(path, mode, 'labels', id+'.txt'), 'a')
            file.write(f"{c} {xcenter} {ycenter} {width} {height}\n")
            file.close()


def main():
    classes = ['accordion', 'cello', 'piano', 'saxophone', 'trumpet', 'violin']
    mapping = {'/m/0mkg': 0, '/m/01xqw': 1, '/m/05r5c': 2, '/m/06ncr': 3, '/m/07gql': 4, '/m/07y_7': 5}
    class_map = {'/m/0mkg': 'accordion', '/m/01xqw': 'cello', '/m/05r5c': 'piano', '/m/06ncr': 'saxophone', '/m/07gql': 'trumpet', '/m/07y_7': 'violin'}

    path = 'oidv6'     # path to current csv files

    for mode in ['train', 'test', 'validation']:
        get_clean_annotation_boxes(path, mode, class_map)
        get_clean_imageids(path, mode)

        # download images using downloader.py and the created imageid-files, then run get_labels
        # get_labels(path, mode, mapping)


if __name__ == '__main__':
    main()
