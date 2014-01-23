#include "BlockController.h"
#include "Metadata.h"
#include "DataManager.h"
#include "FeatureTracker.h"

BlockController::BlockController() {
    blockDim_.x = blockDim_.y = blockDim_.z = 0;
    for (int surface = 0; surface < 6; ++surface) {
        adjacentBlocks_[surface] = SURFACE_NULL;
    }
}

BlockController::~BlockController() {
    pDataManager_->~DataManager();
    pFeatureTracker_->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    initAdjacentBlocks(gridDim, blockIdx);
    
    pDataManager_ = new DataManager();
    pDataManager_->InitTF(meta);
    pDataManager_->LoadDataSequence(meta, gridDim, blockIdx, currentT_);

    pFeatureTracker_ = new FeatureTracker(pDataManager_->GetBlockDim());
    pFeatureTracker_->SetTFRes(pDataManager_->GetTFRes());
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap());
    pFeatureTracker_->SetDataPtr(pDataManager_->GetDataPtr(currentT_));
}

void BlockController::TrackForward(const Metadata& meta, const vec3i& gridDim, const vec3i& blockIdx) {
    pDataManager_->LoadDataSequence(meta, gridDim, blockIdx, currentT_);

    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap());
    pFeatureTracker_->ExtractAllFeatures();
    pFeatureTracker_->TrackFeature(pDataManager_->GetDataPtr(currentT_), FT_FORWARD, FT_DIRECT);
    pFeatureTracker_->SaveExtractedFeatures(currentT_);

    pDataManager_->SaveMaskVolume(pFeatureTracker_->GetMaskPtr(), meta, currentT_);
}

void BlockController::initAdjacentBlocks(const vec3i& gridDim, const vec3i& blockIdx) {
    int px = gridDim.x,   py = gridDim.y,   pz = gridDim.z;
    int x = blockIdx.x,   y = blockIdx.y,   z = blockIdx.z;

    adjacentBlocks_[LEFT]   = x-1 >= 0  ? px*py*z + px*y + x - 1 : -1;
    adjacentBlocks_[RIGHT]  = x+1 <  px ? px*py*z + px*y + x + 1 : -1;
    adjacentBlocks_[BOTTOM] = y-1 >= 0  ? px*py*z + px*(y-1) + x : -1;
    adjacentBlocks_[TOP]    = y+1 <  py ? px*py*z + px*(y+1) + x : -1;
    adjacentBlocks_[FRONT]  = z-1 >= 0  ? px*py*(z-1) + px*y + x : -1;
    adjacentBlocks_[BACK]   = z+1 <  pz ? px*py*(z+1) + px*y + x : -1;
}

std::vector<int> BlockController::GetAdjacentBlocks() {
    std::vector<int> indices;
    for (unsigned int i = 0; i < adjacentBlocks_.size(); ++i) {
        if (adjacentBlocks_[i] != -1) {
            indices.push_back(adjacentBlocks_[i]);
        }
    }
    return indices;
}

void BlockController::UpdateLocalGraph(int blockID, const vec3i& blockIdx) {
    localGraph_.clear();

    std::vector<Feature> *pFeatures = pFeatureTracker_->GetFeatureVectorPtr(currentT_);

    for (unsigned int i = 0; i < pFeatures->size(); ++i) {
        Feature feature = pFeatures->at(i);
        std::vector<int> touchedSurfaces = feature.touchedSurfaces;

        for (unsigned int j = 0; j < touchedSurfaces.size(); ++j) {
            int surface = touchedSurfaces[j];
            int adjacentBlock = adjacentBlocks_[surface];
            if (adjacentBlock == -1) {
                continue;
            }

            vec3i centroid = feature.boundaryCtr[surface];

            Edge edge;
            edge.id         = feature.id;
            edge.start      = blockID;
            edge.end        = adjacentBlock;
            edge.centroid   = centroid + blockDim_ * blockIdx;

            localGraph_.push_back(edge);
        }
    }
}