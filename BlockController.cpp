#include "BlockController.h"

BlockController::BlockController() {}
BlockController::~BlockController() {
    pDataManager_->~DataManager();
    pFeatureTracker_->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata &meta) {
    pDataManager_ = new DataManager();
    pDataManager_->InitTF(meta);
    pDataManager_->LoadDataSequence(meta, currentT_);

    pFeatureTracker_ = new FeatureTracker(pDataManager_->GetBlockDim());
    pFeatureTracker_->SetTFRes(pDataManager_->GetTFRes());
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap());
    pFeatureTracker_->SetDataPtr(pDataManager_->GetDataPtr(currentT_));
}

void BlockController::TrackForward(const Metadata &meta) {
    pDataManager_->LoadDataSequence(meta, currentT_);
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap());
    pFeatureTracker_->ExtractAllFeatures();
    pFeatureTracker_->TrackFeature(pDataManager_->GetDataPtr(currentT_), FT_FORWARD, FT_DIRECT);
    pFeatureTracker_->SaveExtractedFeatures(currentT_);
    pDataManager_->SaveMaskVolume(pFeatureTracker_->GetMaskPtr(), meta, currentT_);
}
