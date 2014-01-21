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
        cout << "cannot load tf setting: " << meta.tfPath() << endl;
        exit(EXIT_FAILURE);
    }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (tfResF < 1) {
        cout << "tfResolution = " << tfResF << endl;
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

    cout << "mask volume created: " << fpath << endl;
}

void DataManager::LoadDataSequence(const Metadata &meta, const int currentT) {
    blockDim_ = meta.volumeDim();
    volumeSize_ = blockDim_.VolumeSize();

    // delete if data is not within [t-2, t+2] of current timestep t
    for (auto it = dataSequence_.begin(); it != dataSequence_.end(); ++it) {
        if (it->first < currentT-2 || it->first > currentT+2) {
            delete [] it->second;
            dataSequence_.erase(it);
            cout << " - " << it->first << endl;
        }
    }

    for (int t = currentT-2; t <= currentT+2; ++t) {
        if (t < meta.start() || t > meta.end() || dataSequence_[t] != NULL) {
            continue;
        }

        char timestamp[21];  // up to 64-bit number
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();

        ifstream inf(fpath.c_str(), ios::binary);
        if (!inf) {
            cout << "cannot read file: " + fpath << endl;
            exit(EXIT_FAILURE);
        }

        float* pData = new float[volumeSize_];
        inf.read(reinterpret_cast<char*>(pData), volumeSize_*sizeof(float));
        inf.close();

        preprocessData(pData);
        dataSequence_[t] = pData;

        cout << " + " << t << endl;
    }
}

void DataManager::preprocessData(float *pData) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize_; ++i) {
        min = std::min(min, pData[i]);
        max = std::max(max, pData[i]);
    }

    cout << min << ", " << max << endl;

    for (int i = 0; i < volumeSize_; ++i) {
        pData[i] = (pData[i] - min) / (max - min);
    }
}
