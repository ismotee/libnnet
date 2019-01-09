#pragma once
#include <vector>
#include <memory>
#include "Neuron.h" // possibly not needed here


enum LinkMethod {
    ALL_COMBINATIONS, EVERY_COMBINATION_OF_TWO, ONE_SELECTED_AND_ONE_RANDOM, RANDOM
};
typedef std::vector<std::vector<int> > LayerLinkIndices, *ptr_LayerLinkIndices;


/**
 * @class NLayer
 * @author Ismo
 * @date 17.04.2018
 * @file libnnet.h
 * @brief Base class for a neural net's layer. Class gives some framework 
 * for managing neurons between upper and lower layer. 
 */

class NLayer {
protected:
    std::shared_ptr<float> bias;
public:
    std::vector<std::shared_ptr<Neuron> > layer;
    void forward();
    void back();
    void setLearningRate(float lr);
    int getLayerSize();
    std::vector<std::shared_ptr<Neuron> >* getLayer();
    void clearErrors();
    std::vector<std::vector<float> > getWeights();
};

class InputLayer : public NLayer {
public:
    InputLayer();
    void link(std::shared_ptr<float> input);
};

class HiddenLayer : public NLayer {
protected:
public:
    HiddenLayer();
    HiddenLayer(int numOfInputs);
    LayerLinkIndices link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndices(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)); // predicate for linker function 
    LayerLinkIndices link(std::shared_ptr<NLayer> upperLayer, LinkMethod METHOD = ALL_COMBINATIONS);
    LayerLinkIndices link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons);
};

class OutputLayer : public NLayer {
public:
    OutputLayer();
    OutputLayer(int numOfInputs);
    LayerLinkIndices link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndices(*method)(std::shared_ptr<NLayer> _upLayer)); // predicate for linker function 
    LayerLinkIndices link(std::shared_ptr<NLayer> upperLayer); // link all
    void back(std::vector<float> desiredOutputs);
};
