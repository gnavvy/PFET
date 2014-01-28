#include "BlockController.h"
#include "Metadata.h"
#include "DataManager.h"
#include "FeatureTracker.h"

BlockController::BlockController() {}

BlockController::~BlockController() {
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    initAdjacentBlocks(gridDim, blockIdx);
    
    pDataManager = new DataManager(meta, gridDim, blockIdx);
    pDataManager->LoadDataSequence(meta, currentT);

    pFeatureTracker = new FeatureTracker(pDataManager->GetBlockDim());
    pFeatureTracker->SetTFRes(pDataManager->GetTFRes());
    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->SetDataPtr(pDataManager->GetDataPtr(currentT));
}

void BlockController::TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    pDataManager->LoadDataSequence(meta, currentT);

    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->ExtractAllFeatures();
    pFeatureTracker->TrackFeature(pDataManager->GetDataPtr(currentT), FT_FORWARD, FT_DIRECT);
    pFeatureTracker->SaveExtractedFeatures(currentT);

    // pDataManager->SaveMaskVolume(pFeatureTracker->GetMaskPtr(), meta, currentT);
    // pDataManager->SaveMaskVolumeMpi(pFeatureTracker->GetMaskPtr(), meta, currentT);
}

void BlockController::initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx) {
    int px = gridDim.x,   py = gridDim.y,   pz = gridDim.z;
    int x = blockIdx.x,   y = blockIdx.y,   z = blockIdx.z;

    adjacentBlocks[LEFT]   = x-1 >= 0  ? px*py*z + px*y + x - 1 : -1;
    adjacentBlocks[RIGHT]  = x+1 <  px ? px*py*z + px*y + x + 1 : -1;
    adjacentBlocks[BOTTOM] = y-1 >= 0  ? px*py*z + px*(y-1) + x : -1;
    adjacentBlocks[TOP]    = y+1 <  py ? px*py*z + px*(y+1) + x : -1;
    adjacentBlocks[FRONT]  = z-1 >= 0  ? px*py*(z-1) + px*y + x : -1;
    adjacentBlocks[BACK]   = z+1 <  pz ? px*py*(z+1) + px*y + x : -1;
}

std::vector<int> BlockController::GetAdjacentBlockIds() {
    std::vector<int> indices;
    for (auto i = 0; i < adjacentBlocks.size(); ++i) {
        if (adjacentBlocks[i] != -1) {
            indices.push_back(adjacentBlocks[i]);
        }
    }
    return indices;
}

void BlockController::UpdateConnectivityTree(int currentBlockId, const vec3i& blockIdx) {
    connectivityTree.clear();

    std::vector<Feature> features = pFeatureTracker->GetFeatures(currentT);
    for (const auto& f : features) {
        for (auto surface : f.touchedSurfaces) {
            int adjacentBlockId = adjacentBlocks[surface];
            if (adjacentBlockId == -1) {
                continue;
            }

            Leaf leaf;
            leaf.id       = f.id;
            leaf.root     = currentBlockId;
            leaf.tip      = adjacentBlockId;
            leaf.centroid = f.boundaryCtr[surface] + blockDim * blockIdx;

            connectivityTree.push_back(leaf);
        }
    }
}