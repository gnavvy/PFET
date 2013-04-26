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
    pFeatureTracker->SetThresholds(LOW_THRESHOLD, HIGH_THRESHOLD);
    pFeatureTracker->SetTFResolution(pDataManager->GetTFResolution());
    pFeatureTracker->SetTFOpacityMap(pDataManager->GetTFOpacityMap());
    pFeatureTracker->SetDataPointer(pDataManager->GetDataPointer(currentTimestep));

}

void BlockController::TrackForward(const Metadata &meta) {
    pDataManager->LoadDataSequence(meta, currentTimestep);
    pFeatureTracker->ExtractAllFeatures();
    pFeatureTracker->TrackFeature(pDataManager->GetDataPointer(currentTimestep), FT_FORWARD, FT_DIRECT);
    pFeatureTracker->SaveExtractedFeatures(currentTimestep);
}
