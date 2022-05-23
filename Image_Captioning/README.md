# Image Captioning
### Image Captioning

In this article,,,,
https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template/blob/master/README.md

- ddd: ddd...

## Installation
Clone the repo and go inside it. Then, run:

```
pip install -r requirements.txt
```

## Structure
```
.
|-- Image_Captioning
|   |-- BEST_checkpoint_5_cap_per_img_.pth.tar
|   |-- Dataset
				|-- MSCOCO_train_val_Korean.json
				|-- TEST_CAPLENS_5_cap_per_img_.json
				|-- TEST_CAPTIONS_5_cap_per_img_.json
				|-- TEST_IMAGES_5_cap_per_img_.hdf5
				|-- TRAIN_CAPLENS_5_cap_per_img_.json
				|-- TRAIN_CAPTIONS_5_cap_per_img_.json
				|-- TRAIN_IMAGES_5_cap_per_img_.hdf5
				|-- VAL_CAPLENS_5_cap_per_img_.json
				|-- VAL_CAPTIONS_5_cap_per_img_.json
				|-- VAL_IMAGES_5_cap_per_img_.hdf5
				|-- test2014
				|-- train2014
				`-- val2014
|   |-- caption.py
|   |-- create_input_files.py
|   |-- datasets.py
|   |-- eval.py
|   |-- models.py
|   |-- train.py
|   `-- utils.py
```

## Project

## Data
create_input_files.py 내의 데이터 경로를 수정하고 실행합니다.
json, hdf5 파일이 생성됩니다. 
```
python create_input_files.py
```


## Models


## Train/Evaluation

train
```
python train.py 
```

eval(This has not yet been implemented.)
```
python eval.py 
```

## Caption
```
python caption.py --img ./Dataset/test2014/COCO_test2014_000000141623.jpg --model ./BEST_checkpoint_5_cap_per_img_.pth.tar
```

## Reference
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning