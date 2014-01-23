#include "MpiController.h"
#include "BlockController.h"
#include "Metadata.h"

MpiController::MpiController() {}

MpiController::~MpiController() {
    pBlockController->~BlockController();
    pMetadata->~Metadata();
    MPI_Finalize();
}

void MpiController::InitWith(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // declare MPI_TYPE_EDGE
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    gridDim = vec3i(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

    blockIdx.z = myRank / (gridDim.x * gridDim.y);
    blockIdx.y = (myRank - blockIdx.z * gridDim.x * gridDim.y) / gridDim.x;
    blockIdx.x = myRank % gridDim.x;

    std::string metaPath(argv[4]);
    pMetadata = new Metadata(metaPath);

    currentT = pMetadata->start();
    
    // init block controller
    pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentT);
    pBlockController->InitParameters(*pMetadata, gridDim, blockIdx);
}

void MpiController::Start() {
    for (int i = pMetadata->start(); i < pMetadata->end(); ++i) {
        TrackForward();
    }

    if (myRank == 0) {
        std::cout << "all jobs done." << std::endl;
    }
}

void MpiController::TrackForward() {
    currentT++;
    if (currentT > pMetadata->end()) {
        currentT = pMetadata->end();
        std::cout << "already last timestep." << std::endl;
        return;
    }

    pBlockController->SetCurrentTimestep(currentT);
    pBlockController->TrackForward(*pMetadata, gridDim, blockIdx);
    pBlockController->UpdateLocalGraph(myRank, blockIdx);

    featureTable.clear();

    adjacentBlocks = pBlockController->GetAdjacentBlocks();
    need_to_send = need_to_recv = true;
    any_send = any_recv = true;

    // while (any_send || any_recv) {
    //     syncFeatureGraph();
    // }

    gatherGlobalGraph();

std::cout << "["<<myRank<<"]" << " #feautre: " << featureTable.size() << std::endl;

    featureTableVector[currentT] = featureTable;
}

void MpiController::gatherGlobalGraph() {
    std::vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    std::vector<int> globalEdgeCountVector(numProc);

    MPI_Allgather(&localEdgeCount, 1, MPI_INT, globalEdgeCountVector.data(), 1, MPI_INT, MPI_COMM_WORLD);

    globalEdgeCount = 0;
    for (unsigned int i = 0; i < globalEdgeCountVector.size(); ++i) {
        globalEdgeCount += globalEdgeCountVector[i];
    }

    std::vector<Edge> globalEdges(globalEdgeCount);

    int displs[numProc];
    displs[0] = 0;
    for (int i = 1; i < numProc; ++i) {
        displs[i] = globalEdgeCountVector[i-1] + displs[i-1];
    }

    MPI_Allgatherv(localEdges.data(), localEdgeCount, MPI_TYPE_EDGE, globalEdges.data(), 
        globalEdgeCountVector.data(), displs, MPI_TYPE_EDGE, MPI_COMM_WORLD);

    mergeCorrespondentEdges(globalEdges);
}

void MpiController::syncFeatureGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> blocksNeedRecv(numProc);
    std::fill(blocksNeedRecv.begin(), blocksNeedRecv.end(), -1);

    int recv_id = need_to_recv ? myRank : -1;
    MPI_Allgather(&recv_id, 1, MPI_INT, blocksNeedRecv.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> adjacentBlocksNeedRecv;

    std::sort(adjacentBlocks.begin(), adjacentBlocks.end());
    std::sort(blocksNeedRecv.begin(), blocksNeedRecv.end());

    std::set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(),
        blocksNeedRecv.begin(), blocksNeedRecv.end(), back_inserter(adjacentBlocksNeedRecv));

    need_to_send = adjacentBlocksNeedRecv.size() > 0 ? true : false;

    std::vector<Edge> adjacentGraph;

    if (need_to_recv) {
        for (unsigned int i = 0; i < adjacentBlocks.size(); ++i) {
            int src = adjacentBlocks.at(i);
            int srcEdgeCount = 0;

            // 1. recv count
            MPI_Irecv(&srcEdgeCount, 1, MPI_INT, src, 100, MPI_COMM_WORLD, &request);
            // 2. recv content
            if (srcEdgeCount != 0) {
                std::vector<Edge> srcEdges(srcEdgeCount);
                MPI_Irecv(srcEdges.data(), srcEdgeCount, MPI_TYPE_EDGE, src, 101, MPI_COMM_WORLD, &request);

                for (int i = 0; i < srcEdgeCount; ++i) {
                    bool isNew = true;
                    for (unsigned int j = 0; j < adjacentGraph.size(); ++j) {
                        if (srcEdges[i] == adjacentGraph[j]) {
                            isNew = false; break;
                        }
                    }
                    if (isNew) {
                        adjacentGraph.push_back(srcEdges[i]);
                    }
                }
            }
        }
        need_to_recv = false;
    }

    if (need_to_send) {
        for (unsigned int i = 0; i < adjacentBlocksNeedRecv.size(); ++i) {
            int dest = adjacentBlocksNeedRecv.at(i);

            // 1. send count
            MPI_Send(&localEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD);
            // 2. send content
            if (localEdgeCount > 0) {
                MPI_Send(localEdges.data(), localEdgeCount, MPI_TYPE_EDGE, dest, 101, MPI_COMM_WORLD);
            }
        }
    }

    // add local edges
    for (unsigned int i = 0; i < localEdges.size(); ++i) {
        bool isNew = true;
        for (unsigned int j = 0; j < adjacentGraph.size(); ++j) {
            if (localEdges[i] == adjacentGraph[j]) {
                isNew = false; break;
            }
        }
        if (isNew) {
            adjacentGraph.push_back(localEdges[i]);
        }
    }

    mergeCorrespondentEdges(adjacentGraph);
    pBlockController->SetLocalGraph(adjacentGraph);

    MPI_Allreduce(&need_to_send, &any_send, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&need_to_recv, &any_recv, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

void MpiController::mergeCorrespondentEdges(vector<Edge> edges) {
    for (unsigned int i = 0; i < edges.size(); ++i) {
        Edge ei = edges[i];
        for (unsigned int j = i+1; j < edges.size(); ++j) {
            Edge ej = edges[j];

            // sync the id of feature if two matches
            if (ei.start == ej.end && ei.end == ej.start &&  // 0->1 | 1->0
                (ei.start == myRank || ei.end == myRank) &&
                ei.centroid.distanceFrom(ej.centroid) <= DIST_THRESHOLD) {
                if (ei.id < ej.id) {    // use the smaller id
                    edges[j].id = ei.id;
                } else {
                    edges[i].id = ej.id;
                }
                updateFeatureTable(ei);
            }
        }
    }

    // if either start or end equals to myRank, add to featureTable
    for (unsigned int i = 0; i < edges.size(); ++i) {
        Edge edge = edges[i];
        if (edge.start == myRank || edge.end == myRank) {
            updateFeatureTable(edge);
        }
    }

    // if both start and end are not equal to myRank,
    // but the id is already in the feature table, update featureTable
    for (unsigned int i = 0; i < edges.size(); ++i) {
        Edge edge = edges[i];
        if (edge.start != myRank || edge.end != myRank) {
            if (featureTable.find(edge.id) != featureTable.end()) {
                updateFeatureTable(edge);
            }
        }
    }

    if (myRank == 0) {     // debug log
        for (auto it = featureTable.begin(); it != featureTable.end(); ++it) {
            int id = it->first;
            cout << "[" << myRank << "]" << id << ": ( ";
            vector<int> value = it->second;
            for (unsigned int i = 0; i < value.size(); ++i) {
                cout << value[i] << " ";
            }
            cout << ")" << endl;
        }
    }
}

void MpiController::updateFeatureTable(Edge edge) {
    if (featureTable.find(edge.id) == featureTable.end()) {
        std::vector<int> value;
        value.push_back(edge.start);
        value.push_back(edge.end);
        featureTable[edge.id] = value;
        need_to_recv = true;
    } else {
        std::vector<int> value = featureTable[edge.id];
        if (find(value.begin(), value.end(), edge.start) == value.end()) {
            value.push_back(edge.start);
            need_to_recv = true;
        }
        if (find(value.begin(), value.end(), edge.end) == value.end()) {
            value.push_back(edge.end);
            need_to_recv = true;
        }
        featureTable[edge.id] = value;
    }
}