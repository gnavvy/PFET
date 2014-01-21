#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "Utils.h"
#include "Metadata.h"

class DataManager {

public:
    DataManager();
   ~DataManager();

    float* GetDataPtr(int t)    { return dataSequence_[t]; }
    float* GetTFMap()           { return pTFMap_; }
    int GetTFRes()              { return tfRes_ > 0 ? tfRes_ : DEFAULT_TF_RES; }
    vector3i GetBlockDim()      { return blockDim_; }

    void InitTF(const Metadata &meta);
    void LoadDataSequence(const Metadata &meta, const int currentT);
    void SaveMaskVolume(float *pData, const Metadata &meta, const int timestep);

private:
    void preprocessData(float *pData);

    DataSequence dataSequence_;
    vector3i blockDim_;

    int volumeSize_;
    int tfRes_;
    float *pTFMap_;
};

#endif // DATAMANAGER_H
