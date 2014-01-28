#ifndef MPICONTROLLER_H
#define MPICONTROLLER_H

#include <fstream>
#include <mpi.h>
#include "Utils.h"

class BlockController;
class Metadata;

class MpiController {
public:
    MpiController();
   ~MpiController();

    void InitWith(int argc, char** argv);
    void Start();

private:
    BlockController *pBlockController;
    Metadata *pMetadata;

    MPI_Datatype MPI_TYPE_LEAF;

    int myRank;
    int numProc;
    int currentT;  // current timestep

    vec3i gridDim;  // #processes in each dimension (xyz)
    vec3i blockIdx; // xyz coordinate of current processor

    // for global graph
    void gatherLeaves();

    // for feature graph
    std::vector<int> adjacentBlocks;
    void syncLeaves();
    void updateFeatureTable(const Leaf& leaf);
    bool toSend, toRecv;
    bool anySend, anyRecv;

    // global feature info
    typedef std::unordered_map<int, std::vector<int> > FeatureTable;
    FeatureTable featureTable;
    std::unordered_map<int, FeatureTable> featureTableVector; // for time varying data

    void mergeCorrespondingEdges(std::vector<Leaf> leaves);
};

#endif // MPICONTROLLER_H