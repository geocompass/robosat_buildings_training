# robosat_buildings_training
使用[mapbox/robosat](https://github.com/mapbox/robosat)训练用来提取建筑物


参考文章：[daniel-j-h : RoboSat ❤️ Tanzania](https://www.openstreetmap.org/user/daniel-j-h/diary/44321)


## 系统准备工作
### 设备及系统
> - 准备一台安装Linux或MacOS系统的机器，可以是CentOS、Ubuntu或MacOS。机器可以是实体机，也可以是VMware虚拟机。

### 安装Docker
> - 在机器中安装Docker，不建议是Windows版Docker。[MacOS安装Docker](https://www.runoob.com/docker/macos-docker-install.html) ，[CentOS安装Docker](https://www.runoob.com/docker/centos-docker-install.html)

### 在Docker中安装Robosat
Robosat 的 [Docker Hub](https://hub.docker.com/r/mapbox/robosat)

可以使用两种方式安装Robosat：
- 使用CPU容器：
```
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu --help
```
- 使用GPU容器（主机上需要 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)：
```
docker run --runtime=nvidia -it --rm -v $PWD:/data --ipc=host mapbox/robosat:latest-gpu train --model /data/model.toml --dataset /data/dataset.toml --workers 4
```

## 数据准备：
### 建筑物轮廓矢量数据
已有的建筑物轮廓矢量数据用来作为建筑物提取的训练数据源。可以有两种方式获取：
- OSM数据源，可以在[geofabrik](http://download.geofabrik.de/)获取，通过[osmium](https://github.com/osmcode/osmium-tool)和[robosat](https://github.com/mapbox/robosat)工具[进行处理](https://www.openstreetmap.org/user/daniel-j-h/diary/44321)。
- 自有数据源。通过[QGIS](https://qgis.org/en/site/)或ArcMAP等工具，加载遥感影像底图，描述的建筑物轮廓Shapefile数据。
本文使用第二种数据来源，并已开源数据源。样例数据覆盖[厦门核心区]()。

### 获取建筑物轮廓geojson数据
> 通过在线工具[mapshaper](https://mapshaper.org/)，将shapefile数据转换为geojson数据。

### 提取训练区覆盖的瓦片行列号
使用robosat的[cover](https://github.com/mapbox/robosat#rs-cover)工具，即可获取当前训练区覆盖的瓦片行列号，并使用csv文件存储。
```
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu cover  --zoom 18 /data/buildings.json /data/buildings.tiles
```
这里是获取在18级下训练区覆盖的瓦片行列号。18级是国内地图通用的最大级别，如果有国外更清晰数据源，可设置更高地图级别。

