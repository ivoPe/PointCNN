# PointCNN

Created by <a href="http://yangyan.li" target="_blank">Yangyan Li</a>,<a href="http://rbruibu.cn" target="_blank"> Rui Bu</a>, <a href="http://www.mcsun.cn" target="_blank">Mingchao Sun</a>, <a href="https://www.weiwu35.com/" target="_blank">Wei Wu</a>, and <a href="http://www.cs.sdu.edu.cn/~baoquan/" target="_blank">Baoquan Chen</a> from Shandong University.

## Installation
A docker file specifies the package used to run the model. To be run on **nvidia-docker** to use GPU.
```bash
cd PointCNN
docker build -t pointcnn .
```

## Tutorial
Once the docker built, you can launch a jupyter notebook:
```bash
# Use nvidia-docker
bash dev.sh
# Launch notebook
bash jupy.sh
```
Main class is **Pcnn_classif** in pointcnn_funcs.py. Check **tuto.ipynb**
