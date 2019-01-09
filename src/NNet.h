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
    std::shared_ptr<InputLayer> inputLayer;
    std::shared_ptr<OutputLayer> outputLayer;
    std::vector<std::shared_ptr<HiddenLayer> > hiddenLayers;

    float curve = 0.82;
    float min = 0.01;
    float max = 0.1;

public:
    NNet();
    NNet(int numOfOutputs, int numOfHiddenLayers);
    void linkInput(std::vector<std::shared_ptr<float> > input);
    LayerLinkIndices linkHidden(int layerDepth, LayerLinkIndices(*method)(NLayer* self, std::shared_ptr<NLayer> _upperLayer));
    LayerLinkIndices linkHidden(int layerDepth, LinkMethod METHOD = ALL_COMBINATIONS);
    LayerLinkIndices linkHidden(int layerDepth, int numOfNeurons);
    LayerLinkIndices linkOutput(LayerLinkIndices(*method)(std::shared_ptr<NLayer> _upperLayer));
    LayerLinkIndices linkOutput(); // Default linker, all neurons from upper layer to every output
    void setLearningcurve(float curve, float min, float max);
    float getLearningrate(int depth);
    std::vector<std::shared_ptr<float> > getOutputSignals();
    std::vector<float> getSums();
    std::vector<std::vector<float> > getWeights(int depth);

    void forward();
    void back(std::vector<float> desiredOut);
    std::string getStats();
};