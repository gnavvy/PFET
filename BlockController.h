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

    std::vector<int> GetAdjacentBlocks();
    std::vector<Edge> GetLocalGraph() { return localGraph; }
    void SetLocalGraph(const std::vector<Edge>& graph) { localGraph = graph; }
    void UpdateLocalGraph(int blockID, const vec3i& blockIdx);

private:
    DataManager    *pDataManager;
    FeatureTracker *pFeatureTracker;
    vec3i           blockDim;
    int             currentT;
    
    std::unordered_map<int, int> adjacentBlocks;
    std::vector<Edge> localGraph;

    void initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx);
};

#endif // BLOCKCONTROLLER_H
