#include "libnnet.h"
#include <iostream>
#include <memory>
#include <cmath>

int lineLength = 100;

int main(int argc, char **argv) {
    std::cout << "making input signals\n";
    std::vector<std::shared_ptr<float> > inputs(3, std::make_shared<float>(0));
    std::vector<std::shared_ptr<float> > outputs;    
    
    std::cout << "creating neural net\n";
    NNet nn = NNet(3, 1);
    std::cout << "creating input neurons\n";
    nn.linkInput(inputs);
    outputs = nn.getOutputs();
    std::cout << "creating input neurons from output\n";
    nn.linkInput(outputs);
    std::cout << "creating hidden layer\n";
    nn.linkHidden(1);
    std::cout << "linking output layer\n";
    nn.linkOutput();
    std::cout << "done!\n";
    std::vector<float> da(3, 0);

    for (int i = 0; i < 200; i++) {
        da[0] = i % 3 == 0 ? 1 : 0;
        da[1] = (i + 1) % 3 == 0 ? 1 : 0;
        da[2] = (i + 2) % 3 == 0 ? 1 : 0;

        std::cout << "stage: " << i % 3 << "\n";

        if (i % 3 == 0) {
            *(inputs[0]) = 1;
            *(inputs[1]) = 0;
            *(inputs[2]) = 0;
        } else if (i % 3 == 1) {
            *inputs[0] = 1;
            *inputs[1] = 1;
            *inputs[2] = 0;
        } else {
            *inputs[0] = 1;
            *inputs[1] = 1;
            *inputs[2] = 1;
        }
        nn.forward();
        nn.back(da);

        std::cout << "input: ";
        for (auto& in : inputs) {
            std::cout << *in << "  ";
        }
        std::cout << "\n";
        std::cout << "output: ";
        for (auto& out : outputs) {
            std::cout << *out << "  ";
        }
        std::cout << "\n";

        std::cout << "desired output: ";
        for (auto& out : da) {
            std::cout << out << "  ";
        }
        std::cout << "\n";
        std::cout << "\n";
    }




    return 0;
}
