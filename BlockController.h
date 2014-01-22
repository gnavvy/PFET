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

    void InitParameters(const Metadata& meta, const Vec3i& gridDim, const Vec3i& blockIdx);
    void TrackForward(const Metadata& meta, const Vec3i& gridDim, const Vec3i& blockIdx);
    void ExtractAllFeatures();
    void SetCurrentTimestep(int t) { currentT_ = t; }

    std::vector<int> GetAdjacentBlocks();
    std::vector<Edge> GetLocalGraph() { return localGraph_; }
    void SetLocalGraph(const std::vector<Edge>& graph) { localGraph_ = graph; }
    void UpdateLocalGraph(int blockID, const Vec3i& blockIdx);

private:
    DataManager    *pDataManager_;
    FeatureTracker *pFeatureTracker_;
    Vec3i           blockDim_;
    int             currentT_;
    
    std::unordered_map<int, int> adjacentBlocks_;
    std::vector<Edge> localGraph_;

    void initAdjacentBlocks(const Vec3i& gridDim, const Vec3i& blockIdx);
};

#endif // BLOCKCONTROLLER_H
