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

    // declare new type for edge
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    gridDim = vec3i(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

    blockIdx.z = myRank / (gridDim.x * gridDim.y);
    blockIdx.y = (myRank - blockIdx.z * gridDim.x * gridDim.y) / gridDim.x;
    blockIdx.x = myRank % gridDim.x;

    std::string metaPath(argv[4]);
    pMetadata = new Metadata(metaPath);

    currentT = pMetadata->start();
    
    pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentT);
    pBlockController->InitParameters(*pMetadata, gridDim, blockIdx);
}

void MpiController::Start() {
    while (currentT < pMetadata->end()) {
        currentT++;
        if (myRank == 0) {
            std::cout << "----- " << currentT << " -----" << std::endl;    
        }

        pBlockController->SetCurrentTimestep(currentT);
        pBlockController->TrackForward(*pMetadata, gridDim, blockIdx);
        pBlockController->UpdateLocalGraph(myRank, blockIdx);

        featureTable.clear();

        adjacentBlocks = pBlockController->GetAdjacentBlockIds();
        need_to_send = need_to_recv = true;
        any_send = any_recv = true;
        while (any_send || any_recv) {
            syncFeatureGraph();
        }

        // gatherGlobalGraph();

        featureTableVector[currentT] = featureTable;
    }

    if (myRank == 0) {
        std::cout << "all jobs done." << std::endl;
    }
}

void MpiController::gatherGlobalGraph() {
    std::vector<Edge> edges = pBlockController->GetLocalGraph();
    int numEdges = edges.size();

    // a vector that holds the number of edges for each block
    std::vector<int> numEdgesGlobal(numProc);
    MPI_Allgather(&numEdges, 1, MPI_INT, numEdgesGlobal.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // be careful, the last argument of std::accumulate controls both value and type
    int numEdgesSum = std::accumulate(numEdgesGlobal.begin(), numEdgesGlobal.end(), 0);
    std::vector<Edge> edgesGlobal(numEdgesSum);

    // where edges received from other blocks should be places.
    std::vector<int> displacements(numProc, 0);
    for (int i = 1; i < numProc; ++i) {
        displacements[i] = numEdgesGlobal[i-1] + displacements[i-1];
    }

    // gather edges from all blocks
    MPI_Allgatherv(edges.data(), numEdges, MPI_TYPE_EDGE, edgesGlobal.data(), 
        numEdgesGlobal.data(), displacements.data(), MPI_TYPE_EDGE, MPI_COMM_WORLD);

    mergeCorrespondingEdges(edgesGlobal);
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
        for (auto i = 0; i < adjacentBlocks.size(); ++i) {
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
        for (auto i = 0; i < adjacentBlocksNeedRecv.size(); ++i) {
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
    for (auto i = 0; i < localEdges.size(); ++i) {
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

    mergeCorrespondingEdges(adjacentGraph);
    pBlockController->SetLocalGraph(adjacentGraph);

    MPI_Allreduce(&need_to_send, &any_send, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&need_to_recv, &any_recv, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

void MpiController::mergeCorrespondingEdges(vector<Edge> edges) {
    for (auto i = 0; i < edges.size(); ++i) {
        Edge ei = edges[i];
        for (auto j = i+1; j < edges.size(); ++j) {
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
    for (auto i = 0; i < edges.size(); ++i) {
        Edge edge = edges[i];
        if (edge.start == myRank || edge.end == myRank) {
            updateFeatureTable(edge);
        }
    }

    // if both start and end are not equal to myRank,
    // but the id is already in the feature table, update featureTable
    for (auto i = 0; i < edges.size(); ++i) {
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
            cout << "[" << myRank << "] " << id << ": ( ";
            vector<int> value = it->second;
            for (auto i = 0; i < value.size(); ++i) {
                cout << value[i] << " ";
            }
            cout << ")" << endl;
        }
    }
}

void MpiController::updateFeatureTable(Edge edge) {
    if (featureTable.find(edge.id) == featureTable.end()) {
        std::vector<int> values;
        values.push_back(edge.start);
        values.push_back(edge.end);
        featureTable[edge.id] = values;
        need_to_recv = true;
    } else {
        std::vector<int> value = featureTable[edge.id];
        if (std::find(value.begin(), value.end(), edge.start) == value.end()) {
            value.push_back(edge.start);
            need_to_recv = true;
        }
        if (std::find(value.begin(), value.end(), edge.end) == value.end()) {
            value.push_back(edge.end);
            need_to_recv = true;
        }
        featureTable[edge.id] = value;
    }
}