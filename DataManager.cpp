#include "Metadata.h"
#include "DataManager.h"

DataManager::DataManager(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    // 1. init transfer function setting
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

    tfRes = static_cast<int>(tfResF);
    tfMap.resize(tfRes);
    inf.read(reinterpret_cast<char*>(tfMap.data()), tfRes*sizeof(float));
    inf.close();

    // 2. init parallel io parameters
    blockDim = meta.volumeDim() / gridDim;
    volumeSize = blockDim.volumeSize();

    int *gsizes = meta.volumeDim().data();
    int *subsizes = blockDim.data();
    int *starts = (blockDim * blockIdx).data();

    MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_FLOAT, &fileType);
    MPI_Type_commit(&fileType);
}

DataManager::~DataManager() {
    MPI_Type_free(&fileType);
}

void DataManager::SaveMaskVolume(float* pData, const Metadata &meta, const int t) {
    char timestamp[21];  // up to 64-bit number
    sprintf(timestamp, (meta.timeFormat()).c_str(), t);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    ofstream outf(fpath.c_str(), ios::binary);
    if (!outf) {
        cerr << "cannot output to file: " << fpath.c_str() << endl;
        exit(EXIT_FAILURE);
    }

    outf.write(reinterpret_cast<char*>(pData), volumeSize*sizeof(float));
    outf.close();

    std::cout << "mask volume created: " << fpath << std::endl;
}

void DataManager::SaveMaskVolumeMpi(float *pData, const Metadata& meta, const int t) {
    // 1. generate output file name
    char timestamp[21];  // up to 64-bit number
    sprintf(timestamp, (meta.timeFormat()).c_str(), t);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    char *cfpath = const_cast<char*>(fpath.c_str());

    // 2. parallel output file
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, cfpath, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0, MPI_FLOAT, fileType, "native", MPI_INFO_NULL);
    MPI_File_write_all(file, dataSequence[t].data(), volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    std::cout << "mask volume created: " << fpath << std::endl;
}

void DataManager::LoadDataSequence(const Metadata& meta, const int currentT) {
    for (auto& data : dataSequence) {
        if (data.first < currentT-2 || data.first > currentT+2) {
            dataSequence.erase(data.first);
        }
    }

    for (int t = currentT-2; t <= currentT+2; ++t) {
        if (t < meta.start() || t > meta.end() || !dataSequence[t].empty()) {
            continue;
        }

        // 1. resize to allocate buffer
        dataSequence[t].resize(volumeSize);

        // 2. generate file name by time step
        char timestamp[21];  // up to 64-bit number
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();
        char *cfpath = const_cast<char*>(fpath.c_str());

        // 3. parallel input file
        MPI_File file;
        int notExist = MPI_File_open(MPI_COMM_WORLD, cfpath, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        if (notExist) {
            std::cout << "cannot read file: " + fpath << std::endl;
            exit(EXIT_FAILURE);
        }
        MPI_File_set_view(file, 0, MPI_FLOAT, fileType, "native", MPI_INFO_NULL);
        MPI_File_read_all(file, dataSequence[t].data(), volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&file);

        // 3. normalize data - parallel
        preprocessData(dataSequence[t]);
    }
}

void DataManager::preprocessData(std::vector<float>& data) {
    float min = data[0], max = data[0];
    for (int i = 1; i < volumeSize; ++i) {
        min = std::min(min, data[i]);
        max = std::max(max, data[i]);
    }

    MPI_Allreduce(&min, &min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max, &max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    for (int i = 0; i < volumeSize; ++i) {
        data[i] = (data[i] - min) / (max - min);
    }
}
