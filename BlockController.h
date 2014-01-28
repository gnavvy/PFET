#ifndef BLOCKCONTROLLER_H
#define BLOCKCONTROLLER_H

#include "Utils.h"

class Metadata;
class DataManager;
class FeatureTracker;

class BlockController {

public:
    BlockController();
   ~BlockController();

    void InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx);
    void TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx);
    void SetCurrentTimestep(int t) { currentT = t; }

    std::vector<int> GetAdjacentBlockIds();
    std::vector<Leaf> GetConnectivityTree() { return connectivityTree; }
    void SetConnectivityTree(const std::vector<Leaf>& tree) { connectivityTree = tree; }
    void UpdateConnectivityTree(int currentBlockId, const vec3i& blockIdx);

private:
    DataManager    *pDataManager;
    FeatureTracker *pFeatureTracker;
    vec3i           blockDim;
    int             currentT;
    
    std::unordered_map<int, int> adjacentBlocks;
    std::vector<Leaf> connectivityTree;

    void initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx);
};

#endif // BLOCKCONTROLLER_H
