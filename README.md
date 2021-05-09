# Object Detection with YOLOv5

This is a deep learning project, exploring object detection using [YOLOv5](https://github.com/ultralytics/yolov5) as the basis.



### Data Set
*Preliminary dataset:* A subset of [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html) containing instrumental classes. The number of images and instances of each class is as follows:

| Class         | Images (Training) | Instances (Training)| Images (Validation) | Instances (Validation)| Images (Testing) | Instances (Testing)|
| ------------- | ----------------- | ------------------- | ------------------- | --------------------- | ---------------- | ------------------ |
| Accordion     | 839               | 955                 | 24                  | 24                    | 77               | 82
| Cello         | 1346              | 2004                | 27                  | 38                    | 78               | 86
| Piano         | 1246              | 1374                | 95                  | 100                   | 267              | 313
| Saxophone     | 854               | 1208                | 33                  | 40                    | 102              | 114
| Trumpet       | 835               | 1546                | 38                  | 65                    | 118              | 172
| Violin        | 1307              | 2028                | 29                  | 36                    | 93               | 101

The dataset labels should be in the following, normalized format:
```
<class> <x-center> <y-center> <width> <height>
```
where ``<class>`` is the index of the class.

There are help functions in ``dataset_utils`` to get and format the data. Download the train, validation and test box annotation csv files from [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html) and use ``cleandata.py`` and ``downloader.py`` to clean the annotation csv files from unnecessary data and download the images for the dataset.


### Experiments
To fine-tune the benchmark model (no frozen layers, no data augmentation):
```
python3 train.py --data ../oidv6/oidv6.yaml --epochs 50 --batch 64 --weights yolov5s.pt --save_period 5 --cache
```

To test a model:
```
python3 test.py --data ../oidv6/oidv6.yaml --weights <path/to/wights.pt> --batch-size 64 --task test --save-txt
```
