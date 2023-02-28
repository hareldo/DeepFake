# DeepFake

EffiecientNet-Bo
![alt text](https://miro.medium.com/max/1400/1*8oE4jOMfOXeEzgsHjSB5ww.png)

Flags:
-m deterime the model type (Origin or Triplet)
-d dataset path (should organized by train,val,test and in each one fake,real folders) for example:

```sh
fakes_dataset
├── train
│   ├── real
│   ├── ├── img1.png
│   ├── ├── img2.png
│   └── fake
│   ├── ├── img1.png
│   ├── ├── img2.png
├── val
│   ├── ├── img1.png
│   ├── ├── img2.png
│   └── fake
│   ├── ├── img1.png
│   ├── ├── img2.png
├── test
│   ├── ├── img1.png
│   ├── ├── img2.png
│   └── fake
│   ├── ├── img1.png
│   ├── ├── img2.png
```

Run original loss:
docker run deep-fake -d C:\workarea\Assignment4_datasets\fakes_datase -m Origin

Run Triplet loss:
docker run deep-fake -d C:\workarea\Assignment4_datasets\fakes_datase -m Triplet