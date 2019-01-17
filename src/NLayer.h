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

    float learningRate;
    int inhibitionInterval;
    float inhibitionTreshold;

    std::vector<std::shared_ptr<Neuron> > layer;
    void forward();
    void back();
    void back(std::vector<float> desiredOutputs); // used by output layer mainly

    // TODO: make only one link and split others into derived classes
    void link(std::shared_ptr<NLayer> upperLayer, void(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)); // predicate for linker function 
    void link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons);
    void link(std::shared_ptr<NLayer> upperLayer);
    void link(std::shared_ptr<float> input);
    void link(std::vector<std::shared_ptr<float> > inputs);

    void setLearningRate(float lr);
    int getLayerSize();
    std::vector<std::shared_ptr<Neuron> >* getLayer();
    std::vector<std::shared_ptr<float> > getOutputSignals();
    void clearErrors();
    std::vector<std::vector<float> > getWeights();
    virtual std::string getType() = 0; // it is a lousy hack
};

class InputLayer : public NLayer {
public:
    InputLayer();
    std::string getType() {return "Input Layer";} // it is a lousy hack
};

class HiddenLayer : public NLayer {
protected:
public:
    HiddenLayer();
    HiddenLayer(int numOfInputs);
    std::string getType() { return "Hidden Layer";} // it is a lousy hack
};

class OutputLayer : public NLayer {
public:
    OutputLayer();
    OutputLayer(int numOfInputs);
    std::string getType() { return "Output Layer";} // it is a lousy hack
};

void linkAll(NLayer* self, std::shared_ptr<NLayer> upperLayer);

