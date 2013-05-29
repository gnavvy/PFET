#include "BlockController.h"

BlockController::BlockController()  {}
BlockController::~BlockController() {
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata &meta) {
    pDataManager = new DataManager();
    pDataManager->LoadDataSequence(meta, currentTimestep);
    pDataManager->CreateNewMaskVolume();
    pDataManager->InitTFSettings(meta.tfPath);

    pFeatureTracker = new FeatureTracker(pDataManager->GetBlockDimension());
    pFeatureTracker->SetTFResolution(pDataManager->GetTFResolution());
    pFeatureTracker->SetTFMap(pDataManager->GetTFOpacityMap());
    pFeatureTracker->SetDataPointer(pDataManager->GetDataPointer(currentTimestep));
}

void BlockController::TrackForward(const Metadata &meta) {
    pDataManager->LoadDataSequence(meta, currentTimestep);
    pFeatureTracker->ExtractAllFeatures();
    pFeatureTracker->TrackFeature(pDataManager->GetDataPointer(currentTimestep), FT_FORWARD, FT_DIRECT);
    pFeatureTracker->SaveExtractedFeatures(currentTimestep);
    pDataManager->SaveMaskVolume(pFeatureTracker->GetMaskPointer(), meta, currentTimestep);
}
