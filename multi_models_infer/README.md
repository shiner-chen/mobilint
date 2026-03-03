# Mobilint Demo

This is a demo using the MLA100.

# Library installations
    $ sudo apt install libopencv-dev

# How to build
    $ make -j
    $ cd build
    $ ./demo

# How to build for aries2
    $ make USE_ARIES2=1 -j
    $ cd build
    $ ./demo

# How to modify the Feeder setting.

- rc/FeederSetting.yaml file needs to be modified.

`FEEDER_TYPE: { CAMERA | VIDEO }`

```
- feeder_type: FEEDER_TYPE
  src_path:
    - Refer to the table below.
- feeder_type:
  src_path:
    - ...
    - ...
- ...
```

| feeder_type |       src_path      | Remarks                                                   |
| -           | -                   | -                                                         |
| CAMERA      | Camera Index        | Enter the camera index as number. ($ls /dev | grep video) |
| VIDEO       | Paths to Video file | Enter the video file path (multiple entries allowed.)     |


# How to modify the Model setting.

- rc/ModelSetting.yaml file needs to be modified.

`MODEL_TYPE: { SSD | POSE | FACENET | STYLENET | SEGMENTATION }`

`CLUSTER: { Cluster0 | Cluster1 }`

`CORE: { Core0 | Core1 | Core2 | Core3 }`

```
- model_type: MODEL_TYPE
  mxq_path: MXQ file path
  dev_no: Index of the MLA100 board.
  core_id:
    - cluster: CLUSTER
      core: CORE
    - cluster: CLUSTER
      core: CORE
    - ...
- model_type:
  mxq_path:
  dev_no:
  core_id:
    - ...
- ...
```

# Shortcut

| Shortcut | Description            |
| ---      | ---                    |
| D, d     | Display FPS            |
| M, m     | Maximize Screen        |
| C, c     | Clear All Worker       |
| F, f     | Fill All Worker        |
| T, t     | Display Running Time   |
| Q, q     | Quit                   |
