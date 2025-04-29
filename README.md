# FiGO and TASM\* integration in VDBMS

## All major code changes are provided in *`'figo_newds.patch'`* and *`'tasm_patch_withFixforRTX30series.patch'`* files for the successful integration.

```
git clone https://github.com/chirag26495/figo_tasmST.git
cd figo_tasmST/
git clone https://github.com/uwdb/TASM.git
cd TASM
git submodule init
git submodule update
cp ../tasm_patch_withFixforRTX30series.patch .
git apply tasm_patch_withFixforRTX30series.patch
cd python/Examples/
git clone https://github.com/jiashenC/FiGO.git
cd FiGO/
### to git home dir
cd ../../../../
cp allObjQueries_FiGO_TASMst.py  figo_newds.patch  figo-tasm_video-object_query.py  TASM/python/Examples/FiGO/
cd TASM/python/Examples/FiGO/
git apply figo_newds.patch
cd weights/
python3 get_weights.py

### to TASM dir
cd ../../../
docker build -t tasm/environment -f docker/Dockerfile.environment  .
docker build -t tasm/tasm -f docker/Dockerfile .
docker run -it --runtime=nvidia -p 8890:8890 --name tasm tasm/tasm:latest /bin/bash

#### Figo dependencies
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install webcolors==1.11.1
sudo apt-get install python3-tk

cd FiGO/
python3 figo-tasm_video-object_query.py --video ../data/birds.mp4 --label bird --labelcount 1 --out query_out/
python3 allObjQueries_FiGO_TASMst.py --videopath  ../data/birds.mp4  --startlabel  bird --queryoutdir  query_out
```

## Run on your custom video

1. Convert the video file format to HEVC and ensure its dimensions are divisible by *32* for its compatibility with TASMst
```
chomd +x convert_video.sh
./convert_video.sh input.mp4 output.mp4
```

2. Run script for Automated retrieval and profiling of all object categories present in your video
```
python3 allObjQueries_FiGO_TASMst.py --videopath  output.mp4  --startlabel  seed_object_type --queryoutdir  output_dir
```
3. Get all retrieved objects' data statistics along with their query times from `output_dir`

4. Run script for querying a single object using the cached semantic index of FiGO and TASM integrated VideoDBMS
```
python3 figo-tasm_video-object_query.py --video output.mp4 --label query_object_type --labelcount 1 --out query_out/
```
5. Get a single object's cropped JPGs and queried attributes in CSV in `query_out` directory


## References

Thanks to the amazing open-source github repositories:

**[TASM](https://github.com/uwdb/TASM)**

**[FiGO](https://github.com/jiashenC/FiGO)**
