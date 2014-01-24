#include "BlockController.h"
#include "Metadata.h"
#include "DataManager.h"
#include "FeatureTracker.h"

BlockController::BlockController() {
    for (int surface = 0; surface < 6; ++surface) {
        adjacentBlocks[surface] = SURFACE_NULL;
    }
}

BlockController::~BlockController() {
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    initAdjacentBlocks(gridDim, blockIdx);
    
    pDataManager = new DataManager(meta);
    pDataManager->LoadDataSequence(meta, gridDim, blockIdx, currentT);

    pFeatureTracker = new FeatureTracker(pDataManager->GetBlockDim());
    pFeatureTracker->SetTFRes(pDataManager->GetTFRes());
    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->SetDataPtr(pDataManager->GetDataPtr(currentT));
}

void BlockController::TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    pDataManager->LoadDataSequence(meta, gridDim, blockIdx, currentT);

    pFeatureTracker->SetTFMap(pDataManager->GetTFMap());
    pFeatureTracker->ExtractAllFeatures();
    pFeatureTracker->TrackFeature(pDataManager->GetDataPtr(currentT), FT_FORWARD, FT_DIRECT);
    pFeatureTracker->SaveExtractedFeatures(currentT);

    pDataManager->SaveMaskVolume(pFeatureTracker->GetMaskPtr(), meta, currentT);
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

std::vector<int> BlockController::GetAdjacentBlocks() {
    std::vector<int> indices;
    for (unsigned int i = 0; i < adjacentBlocks.size(); ++i) {
        if (adjacentBlocks[i] != -1) {
            indices.push_back(adjacentBlocks[i]);
        }
    }
    return indices;
}

void BlockController::UpdateLocalGraph(int blockID, const vec3i& blockIdx) {
    localGraph.clear();

    std::vector<Feature> *pFeatures = pFeatureTracker->GetFeatureVectorPtr(currentT);

    for (unsigned int i = 0; i < pFeatures->size(); ++i) {
        Feature feature = pFeatures->at(i);
        std::vector<int> touchedSurfaces = feature.touchedSurfaces;

        for (unsigned int j = 0; j < touchedSurfaces.size(); ++j) {
            int surface = touchedSurfaces[j];
            int adjacentBlock = adjacentBlocks[surface];
            if (adjacentBlock == -1) {
                continue;
            }

            vec3i centroid = feature.boundaryCtr[surface];

            Edge edge;
            edge.id         = feature.id;
            edge.start      = blockID;
            edge.end        = adjacentBlock;
            edge.centroid   = centroid + blockDim * blockIdx;

            localGraph.push_back(edge);
        }
    }
}