#include "BlockController.h"

int main (int argc, char** argv) {

    Metadata meta; {
        meta.start      = 60;
        meta.end        = 65;
        meta.prefix     = "vorts";
        meta.surfix     = "data";
        meta.path       = "/Users/Yang/Develop/Data/vorts";
        meta.tfPath     = "/Users/Yang/Develop/Data/vorts/vorts.tfe";
        meta.timeFormat = "%d";
        meta.volumeDim  = Vector3i(128, 128, 128);
    }

    int currentTimestep = meta.start;

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(meta);

    while (currentTimestep++ < meta.end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(meta);
        cout << "-- " << currentTimestep << " done --" << endl;
    }

    delete pBlockController;
    return 0;
}
