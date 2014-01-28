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
    MPI_Type_contiguous(sizeof(Leaf), MPI_BYTE, &MPI_TYPE_LEAF);
    MPI_Type_commit(&MPI_TYPE_LEAF);

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
        pBlockController->UpdateConnectivityTree(myRank, blockIdx);

        featureTable.clear();

        adjacentBlocks = pBlockController->GetAdjacentBlockIds();
        toSend = toRecv = anySend = anyRecv = true;
        while (anySend || anyRecv) {
            syncLeaves();
        }

        // gatherLeaves();

        if (myRank == 0) {
            for (const auto& f : featureTable) {
                std::cout << "["<<myRank<<"] " << f.first << ":";
                for (const auto& id : f.second) {
                    std::cout << id << " ";
                }
                std::cout << std::endl;
            }            
        }

        featureTableVector[currentT] = featureTable;
    }

    if (myRank == 0) {
        std::cout << "all jobs done." << std::endl;
    }
}

void MpiController::gatherLeaves() {
    std::vector<Leaf> leaves = pBlockController->GetConnectivityTree();
    int numLeaves = leaves.size();

    // a vector that holds the number of leaves for each block
    std::vector<int> numLeavesGlobal(numProc);
    MPI_Allgather(&numLeaves, 1, MPI_INT, numLeavesGlobal.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // be careful, the last argument of std::accumulate controls both value and type
    int numLeavesSum = std::accumulate(numLeavesGlobal.begin(), numLeavesGlobal.end(), 0);
    std::vector<Leaf> leavesGlobal(numLeavesSum);

    // where leaves received from other blocks should be places. 
    std::vector<int> displacements(numProc, 0);
    for (int i = 1; i < numProc; ++i) {
        displacements[i] = numLeavesGlobal[i-1] + displacements[i-1];
    }

    // gather leaves from all blocks
    MPI_Allgatherv(leaves.data(), numLeaves, MPI_TYPE_LEAF, leavesGlobal.data(),
        numLeavesGlobal.data(), displacements.data(), MPI_TYPE_LEAF, MPI_COMM_WORLD);

    mergeCorrespondingEdges(leavesGlobal);
}

void MpiController::syncLeaves() {
    std::vector<Leaf> myLeaves = pBlockController->GetConnectivityTree();
    int numLeaves = myLeaves.size();

    int recvId = toRecv ? myRank : -1;
    std::vector<int> blocksNeedRecv(numProc, -1);   
    MPI_Allgather(&recvId, 1, MPI_INT, blocksNeedRecv.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> adjacentBlocksNeedRecv;

    // find if any of my neighbors need to receive
    std::sort(adjacentBlocks.begin(), adjacentBlocks.end());
    std::sort(blocksNeedRecv.begin(), blocksNeedRecv.end());
    std::set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(),
        blocksNeedRecv.begin(), blocksNeedRecv.end(), back_inserter(adjacentBlocksNeedRecv));

    // no bother to send if nobody need to receive
    toSend = adjacentBlocksNeedRecv.empty() ? false : true;

    if (toRecv) {
        for (auto neighbor : adjacentBlocks) {
            int numLeaves = 0;
            MPI_Request request;
            // 1. see how many leaves my neighbor have
            MPI_Irecv(&numLeaves, 1, MPI_INT, neighbor, 100, MPI_COMM_WORLD, &request);
            // 2. if they have any, get them
            if (numLeaves != 0) {
                std::vector<Leaf> leaves(numLeaves);
                MPI_Irecv(leaves.data(), numLeaves, MPI_TYPE_LEAF, neighbor, 101, MPI_COMM_WORLD, &request);
                for (const auto& leaf : leaves) {
                    // 3. if I don't previously have the leaf my neighbor send to me, take it
                    if (std::find(myLeaves.begin(), myLeaves.end(), leaf) == myLeaves.end()) {
                        myLeaves.push_back(leaf);
                    }
                }
            }
        }
        toRecv = false;
    }

    if (toSend) {
        for (auto neighbor : adjacentBlocksNeedRecv) {
            // 1. tell them how many leaves I have, even if I have none
            MPI_Send(&numLeaves, 1, MPI_INT, neighbor, 100, MPI_COMM_WORLD);
            // 2. if I have any, send them
            if (numLeaves > 0) {
                MPI_Send(myLeaves.data(), numLeaves, MPI_TYPE_LEAF, neighbor, 101, MPI_COMM_WORLD);
            }
        }
    }

    mergeCorrespondingEdges(myLeaves);
    pBlockController->SetConnectivityTree(myLeaves);

    // anySend = any(toSend), anyRecv = any(toRecv)
    MPI_Allreduce(&toSend, &anySend, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&toRecv, &anyRecv, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

void MpiController::mergeCorrespondingEdges(std::vector<Leaf> leaves) {
    for (auto i = 0; i < leaves.size(); ++i) {
        Leaf& p = leaves[i];
        for (auto j = i+1; j < leaves.size(); ++j) {
            Leaf& q = leaves[j];

            // sync leaf id of feature with the smaller one if two match
            if (p.root == q.tip && p.tip == q.root && (p.root == myRank || p.tip == myRank) &&
                p.centroid.distanceFrom(q.centroid) < DIST_THRESHOLD) {
                p.id = q.id = std::min(p.id, q.id);
                // updateFeatureTable(leaves[i]);
            }
        }
    }

    // if either root or tip equals to myRank, add to featureTable
    // if both root and tip are not equal to myRank,
    // but the id is already in the feature table, update featureTable
    for (const auto& leaf : leaves) {
        if (leaf.root == myRank || leaf.tip == myRank || featureTable.find(leaf.id) != featureTable.end()) {
            updateFeatureTable(leaf);
        }
    }
}

void MpiController::updateFeatureTable(const Leaf& leaf) {
    if (featureTable.find(leaf.id) == featureTable.end()) {
        std::vector<int> values;
        values.push_back(leaf.root);
        values.push_back(leaf.tip);
        featureTable[leaf.id] = values;
        toRecv = true;
    } else {
        std::vector<int> &value = featureTable[leaf.id];
        if (std::find(value.begin(), value.end(), leaf.root) == value.end()) {
            value.push_back(leaf.root);
            toRecv = true;
        }
        if (std::find(value.begin(), value.end(), leaf.tip) == value.end()) {
            value.push_back(leaf.tip);
            toRecv = true;
        }
    }
}