#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Utils.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitParameters(const Metadata &meta);
    void TrackForward(const Metadata &meta);
    void ExtractAllFeatures();
    void SetCurrentTimestep(int t) { currentTimestep = t; }

private:
    DataManager             *pDataManager;
    FeatureTracker          *pFeatureTracker;
    FeatureVectorSequence    featureSequence;
    int                      currentTimestep;
};

#endif // DATABLOCKCONTROLLER_H
