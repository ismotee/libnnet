#include "libnnet.h"
#include <iostream>
#include <memory>
#include <cmath>

int lineLength = 100;

int main(int argc, char **argv) {
    std::cout << "making input signals\n";
    std::vector<std::shared_ptr<float> > inputs;
    std::vector<std::shared_ptr<float> > outputs;
    inputs.push_back(std::make_shared<float>(0));
    inputs.push_back(std::make_shared<float>(0));

    NNet nn = NNet(2, 2, 1);
    nn.linkInput(inputs);
    nn.linkHidden(1);
    nn.linkOutput();

    outputs = nn.getOutputs();

    std::vector<float> da (2,0);
    
    for (int i = 0; i < 200; i++) {
        nn.forward();
        nn.back(da);

        std::cout << "output: ";
        for (auto& out : outputs) {
            std::cout << *out << "  ";
        }
        std::cout << "\n";
    }




    return 0;
}
