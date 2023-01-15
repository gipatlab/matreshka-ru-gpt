# matreshka-ru-gpt

``` 
export LD_LIBRARY_PATH=/usr/lib/
```

```
apt-get install clang-9 llvm-9 llvm-9-dev llvm-9-tools
```

```
rm -rf apex
```

```
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```
pip install triton==1.0.0
```

```
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed
```

```
ds_report
```

```
rm -rf ru-gpts
```

```
git clone  https://github.com/sberbank-ai/ru-gpts
```

```
pip install transformers
```

```
pip install huggingface_hub
```

```
pip install timm==0.3.2
```

```
cp ru-gpts/src_utils/trainer_pt_utils.py /usr/local/lib/python3.8/dist-packages/transformers/trainer_pt_utils.py
```

```
cp ru-gpts/src_utils/_amp_state.py /usr/local/lib/python3.8/dist-packages/apex/amp/_amp_state.py
```