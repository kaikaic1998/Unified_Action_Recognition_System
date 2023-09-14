import json

annotation_file_path = '../COCO/annotations/person_keypoints_train2017.json'
annotation = json.load(open(annotation_file_path))
print(annotation)
