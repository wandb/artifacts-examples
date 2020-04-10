set -e

cd demodata/coco 

curl http://images.cocodataset.org/zips/val2017.zip > ./val2017.zip
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip > ./annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip
