#include "BlockController.h"
#include "Metadata.h"

using namespace std;

int main () {
    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/vorts.config");
    int currentT = meta.start();

    BlockController blockController;
    blockController.SetCurrentTimestep(currentT);
    blockController.InitParameters(meta);

    while (currentT < meta.end()) {
        blockController.SetCurrentTimestep(currentT);
        blockController.TrackForward(meta);
        currentT++;
        cout << "-- " << currentT << " done --" << endl;
    }

    return EXIT_SUCCESS;
}
