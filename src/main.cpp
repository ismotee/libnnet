#include "libnnet.h"
#include <iostream>
#include <memory>
#include <cmath>

const int NUM_OF_INPUTS = 2;
const int NUM_OF_OUTPUTS = 3;
const int NUM_OF_HIDDEN_NEURONS = 20;

int main(int argc, char **argv) {
    std::cout << "making input signals\n";
    std::vector<std::shared_ptr<float> > inputs;
    std::vector<std::shared_ptr<float> > outputs;    

    for (int i = 0; i < NUM_OF_INPUTS; i++) {
        inputs.push_back(std::make_shared<float>(0));
    }

    NNet nn;
    std::vector<float> douts(NUM_OF_OUTPUTS, 0);

    nn.createInputLayerAndLink(inputs);
    nn.createHiddenLayerAndLink(NUM_OF_HIDDEN_NEURONS);
    outputs = nn.createOutputLayerAndLink(NUM_OF_OUTPUTS);
    nn.printStats();
    
    for (int i = 0; i < 4000; i++) {
        douts[0] = i % 4 == 0 ? 1 : 0;
        douts[1] = (i + 1) % 4 == 0 ? 1 : 0;
        douts[2] = (i + 2) % 4 == 0 || (i + 3) % 4 == 0 ? 1 : 0;

        std::cout << "stage: " << i % 4 << "\n";

        if (i % 4 == 0) {
            douts[0] = 1;
            douts[1] = 0;
            douts[2] = 0;
            *(inputs[0]) = 1;
            *(inputs[1]) = 0;
        } else if (i % 4 == 1) {
            douts[0] = 0;
            douts[1] = 0;
            douts[2] = 1;
            *inputs[0] = 0;
            *inputs[1] = 1;
        } else if (i % 4 == 2) {
            douts[0] = 0;
            douts[1] = 1;
            douts[2] = 0;
            *inputs[0] = 0;
            *inputs[1] = 0;
        } else {
            douts[0] = 0;
            douts[1] = 1;
            douts[2] = 0;
            *inputs[0] = 1;
            *inputs[1] = 1;
        }
        nn.forward();
        nn.back(douts);
        std::cout << "input: ";
        for (auto in : inputs) {
            std::cout << *(in) << "  ";
        }
        std::cout << "\n";
        std::cout << "output: ";
        for (auto out : outputs) {
            std::cout << *(out) << "  ";
        }
        std::cout << "\n";

        std::cout << "desired output: ";
        for (auto dout : douts) {
            std::cout << dout << "  ";
        }
        std::cout << "\n";
        std::cout << "\n";
    }

    return 0;
}
