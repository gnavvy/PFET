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
    void TrackForward();

private:
    BlockController *pBlockController;
    Metadata *pMetadata;

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Comm MPI_COMM_LOCAL;
    MPI_Request request;
    MPI_Status status;

    int myRank;
    int numProc;
    int currentT;  // current timestep

    vec3i gridDim;  // #processes in each dimension (xyz)
    vec3i blockIdx; // xyz coordinate of current processor

    // for global graph
    int globalEdgeCount;
    void gatherGlobalGraph();

    // for feature graph
    std::vector<int> adjacentBlocks;
    void syncFeatureGraph();
    void updateFeatureTable(Edge edge);
    bool need_to_send;
    bool need_to_recv;
    bool any_send, any_recv;

    // global feature info
    typedef std::unordered_map<int, std::vector<int> > FeatureTable;
    FeatureTable featureTable;
    std::unordered_map<int, FeatureTable> featureTableVector; // for time varying data

    void mergeCorrespondentEdges(std::vector<Edge> edges);
};

#endif // MPICONTROLLER_H