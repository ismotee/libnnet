#include <memory>
#include <vector>
#include <iostream>

#include "Input.h"
#include "Neuron.h"
#include "NLayer.h"

/**
 * @class NNet
 * @author Ismo
 * @date 07.05.2018
 * @file libnnet.h
 * @brief General neural net class. NNet has one input, one output layer and
 * dynamic amount of hidden layers.		 
 */

class NNet {
    std::vector<std::shared_ptr<NLayer> > layers;

public:
    NNet();
    void createInputLayerAndLink(std::vector<std::shared_ptr<float> > input);
    void createHiddenLayerAndLink(int numOfNeurons);
    std::vector<std::shared_ptr<float> > createOutputLayerAndLink(int numOfOutputs);
    void setLearningcurve(float curve, float min, float max);
    float getLearningrate(int depth);
    std::vector<std::shared_ptr<float> > getOutputSignals();

    void forward();
    void back(std::vector<float> desiredOut);
    void printStats();
};