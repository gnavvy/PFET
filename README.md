### PFET -- Parallel Feature Extraction and Tracking
====

#### Dependency

1. open-mpi 1.7.x or above
2. a modern compiler with C++11 support

#### Installation

PFET uses cmake to build the base system. After downloading the source code, create a new ``build`` directory inside the source directory and run:

    cmake ../CMakeFiles.txt

or 

    ccmake ..

if you prefer to use the GUI. 

After configuration, simply run

    make

to generate the executable file.

#### Basic Usage

Assume the MPI libraries are properly installed, use the following script in to run the program in parallel.

    mpirun -np X ./<program> <gridDim.x> <gridDim.y> <gridDim.z> ../<config_file_name>
    
For example:

    mpirun -np 8 ./pfet 2 2 2 ../ball.config
    
will divide the domain into 8 (2 x 2 x 2) subdomains and handle them in parallel.

#### Configuration File

The last argument of the above script point to a text based configuration file, in which detailed information about the data can be configured.

| Parameter  | Type      | Description                          | Example         | 
|:---------- |:--------- |:------------------------------------ | :-------------- |
| start      | int       | Starting index of the file sequence  | 0               |
| end        | int       | Ending index of a file sequence      | 4               |
| prefix     | string    | Prefix of the file name              | "ball_"         |
| suffix     | string    | Suffix of the file name              | "raw"           |
| path       | string    | Path to the data files ``directory`` | "path/to/dir"   |
| tfPath     | string    | Path to transfer function setting    | "path/to/file"  |
| timeFormat | string    | String formatting for int->string    | "%3d"           |
| volumeDim  | list(int) | Volume dimension                     | (128, 128, 128) |

For example, the following data files will be read in with the above settings:

    path/to/file/ball_000.raw
    path/to/file/ball_001.raw
    path/to/file/ball_002.raw
    path/to/file/ball_003.raw
    path/to/file/ball_004.raw

#### Related Publication

Please consider citing the following reference if you find the code helpful.

Yang Wang, Hongfeng Yu, Kwan-Liu Ma. *Scalable Parallel Feature Extraction and Tracking for Large Time-varying 3D Volume Data*. In Proceedings of EGPGV 2013, pages 17-24.

###### BibTex

    @article{wang2013scalable,
      title={Scalable Parallel Feature Extraction and Tracking for Large Time-varying 3D Volume Data},
      author={Wang, Yang and Yu, Hongfeng and Ma, Kwan-Liu},
      booktitle={Eurographics Symposium on Parallel Graphics and Visualization},
      pages={17--24},
      year={2013},
      organization={The Eurographics Association}
    }

#### Acknowledgments

This software is made available by the VIDI research group at the University of 
California at Davis through a project sponsored by the U.S. Department of Energy 
with grants DE-FC02-06ER25777, DE-CS0005334, and DE-FC02-2ER26072.

#### License
MIT