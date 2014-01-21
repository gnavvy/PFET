#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Utils.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
   ~BlockController();

    void InitParameters(const Metadata& meta);
    void TrackForward(const Metadata& meta);
    void ExtractAllFeatures();
    void SetCurrentTimestep(int t) { currentT_ = t; }

private:
    DataManager    *pDataManager_;
    FeatureTracker *pFeatureTracker_;
    int             currentT_;
};

#endif // DATABLOCKCONTROLLER_H
