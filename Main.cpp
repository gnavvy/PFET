#include <iostream>
#include "MpiController.h"

int main (int argc, char **argv) {
    if (argc != 5) {
        std::cout << "argv[0]: " << argv[0] << std::endl;
        std::cout << "argv[1]: " << argv[1] << std::endl;
        std::cout << "argv[2]: " << argv[2] << std::endl;
        std::cout << "argv[3]: " << argv[3] << std::endl;
        std::cout << "argv[4]: " << argv[4] << std::endl;
        std::cout << "Usage : " << argv[0] << " npx npy npz" << "argv[4]" << std::endl;
        return EXIT_FAILURE;
    }

    MpiController mc;
    mc.InitWith(argc, argv);
    mc.Start();

    return EXIT_SUCCESS;
}
