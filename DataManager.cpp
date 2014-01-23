#include <mpi.h>
#include "Metadata.h"
#include "DataManager.h"

DataManager::DataManager() {}

DataManager::~DataManager() {
    if (!dataSequence_.empty()) {
        for (auto it = dataSequence_.begin(); it != dataSequence_.end(); ++it) {
            delete [] it->second;  // unload data
        }
    }
}

void DataManager::InitTF(const Metadata &meta) {
    ifstream inf(meta.tfPath().c_str(), ios::binary);
    if (!inf) {
        std::cout << "cannot load tf setting: " << meta.tfPath() << std::endl;
        exit(EXIT_FAILURE);
    }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (tfResF < 1) {
        std::cout << "tfResolution = " << tfResF << std::endl;
        exit(EXIT_FAILURE);
    }

    tfRes_ = (int)tfResF;
    pTFMap_ = new float[tfRes_];
    inf.read(reinterpret_cast<char*>(pTFMap_), tfRes_*sizeof(float));
    inf.close();
}

void DataManager::SaveMaskVolume(float* pData, const Metadata &meta, const int timestep) {
    char timestamp[21];  // up to 64-bit number
    sprintf(timestamp, (meta.timeFormat()).c_str(), timestep);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    ofstream outf(fpath.c_str(), ios::binary);
    if (!outf) {
        cerr << "cannot output to file: " << fpath.c_str() << endl;
        exit(EXIT_FAILURE);
    }

    outf.write(reinterpret_cast<char*>(pData), volumeSize_*sizeof(float));
    outf.close();

    std::cout << "mask volume created: " << fpath << std::endl;
}

void DataManager::LoadDataSequence(const Metadata& meta, const vec3i& gridDim, 
    const vec3i& blockIdx, const int currentT) {

    blockDim_ = meta.volumeDim() / gridDim;
    volumeSize_ = blockDim_.volumeSize();

    // delete if data is not within [t-2, t+2] of current timestep t
    for (auto it = dataSequence_.begin(); it != dataSequence_.end(); ++it) {
        if (it->first < currentT-2 || it->first > currentT+2) {
            delete [] it->second;
            dataSequence_.erase(it);
        }
    }

    for (int t = currentT-2; t <= currentT+2; ++t) {
        if (t < meta.start() || t > meta.end() || dataSequence_[t] != nullptr) {
            continue;
        }

        // 1. allocate new data buffer
        dataSequence_[t] = new float[volumeSize_];

        // 2. generate file name by timestep
        char timestamp[21];  // up to 64-bit number
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();
        char *cfpath = const_cast<char*>(fpath.c_str());

        // 3. parallel io
        int *gsizes = meta.volumeDim().data();
        int *subsizes = blockDim_.data();
        int *starts = (blockDim_ * blockIdx).data();

        MPI_Datatype filetype;
        MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_FLOAT, &filetype);
        MPI_Type_commit(&filetype);

        MPI_File file;
        int notExist = MPI_File_open(MPI_COMM_WORLD, cfpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        if (notExist) {
            std::cout << "cannot read file: " + fpath << std::endl;
            exit(EXIT_FAILURE);
        }

        MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
        MPI_File_read_all(file, dataSequence_[t], volumeSize_, MPI_FLOAT, MPI_STATUS_IGNORE);

        // 3. gc
        MPI_File_close(&file);
        MPI_Type_free(&filetype);

        // 4. nomalize data - parallel
        preprocessData(dataSequence_[t]);
    }
}

void DataManager::preprocessData(float *pData) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize_; ++i) {
        min = std::min(min, pData[i]);
        max = std::max(max, pData[i]);
    }

    MPI_Allreduce(&min, &min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max, &max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    for (int i = 0; i < volumeSize_; ++i) {
        pData[i] = (pData[i] - min) / (max - min);
    }
}
