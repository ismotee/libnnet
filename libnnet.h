#pragma once
#include <vector>
#include <memory>

enum LinkMethod {
    ALL_COMBINATIONS, EVERY_COMBINATION_OF_TWO, ONE_SELECTED_AND_ONE_RANDOM, RANDOM
};
typedef std::vector<std::vector<int> > LayerLinkIndexes, *ptr_LayerLinkIndexes;
/**
 * @class Input
 * @author Ismo Torvinen
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Input is class for Neuron inputs. Input stores and manages weights and 
 * Input is capable of calculating back propagation from neuron base delta values.
 */
class Neuron;

class Input {
    std::shared_ptr<float> in;
    std::shared_ptr<Neuron> linkedNeuron;
    float weight;
public:
    Input();
    Input(std::shared_ptr<float> in_);
    Input(std::shared_ptr<Neuron> neuron);
    // input control methods
    void connect(Input& in_);
    void connect(std::shared_ptr<Neuron> neuron);
    void connect(std::shared_ptr<float> in_);
    void disconnect();
    // weight control methods
    float getWeight();
    float getWeightedInput();
    void setWeight(float w);
    std::shared_ptr<float> getInput();
    std::shared_ptr<Neuron> getNeuron();
};

/**
 * @class Neuron
 * @author Ismo
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Neuron with dynamic amount of inputs and one output. Neuron stores the brief history of Neuron activity.
 * Neuron takes care of inhibition as well as data that is needed for back propagation training of Input.
 */


class Neuron {
    float learningRate;
    std::vector<Input> inputs;
    float sum;
    std::shared_ptr<float> outputSignal;

public:
    float error;
    Neuron(float bias = -1);
    Neuron(std::vector<std::shared_ptr<float> > ins_, float bias = -1);
    void addInput(std::shared_ptr<float> out);
    void addInput(std::shared_ptr<Neuron> neuron);
    void addInputs(std::vector<std::shared_ptr<Neuron> > ins_);
    void forward();
    void back(float desiredOut);
    void back();
    // getters and setters
    std::shared_ptr<float> getOutputSignal();
    float getSum();
    float getCurrentError();
    void setLearningRate(float lr);
    std::vector<float> getInputValues();
    std::vector<float> getWeights();

};

/**
 * @class nLayer
 * @author Ismo
 * @date 17.04.2018
 * @file libnnet.h
 * @brief Base class for a neural net's layer. Class gives some framework 
 * for managing neurons between upper and lower layer. 
 */

class NLayer {
protected:

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
    LayerLinkIndexes link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndexes(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)); // predicate for linker function 
    LayerLinkIndexes link(std::shared_ptr<NLayer> upperLayer, LinkMethod METHOD = ALL_COMBINATIONS);
    LayerLinkIndexes link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons);
};

class OutputLayer : public NLayer {
public:
    OutputLayer();
    OutputLayer(int numOfInputs);
    void link(std::shared_ptr<NLayer> upperLayer, void* (*method)(std::shared_ptr<NLayer> _upLayer)); // predicate for linker function 
    void link(std::shared_ptr<NLayer> upperLayer); // link all
    void back(std::vector<float> desiredOutputs);
};

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
    LayerLinkIndexes linkHidden(int layerDepth, LayerLinkIndexes(*method)(NLayer* self, std::shared_ptr<NLayer> _upperLayer));
    LayerLinkIndexes linkHidden(int layerDepth, LinkMethod METHOD = ALL_COMBINATIONS);
    LayerLinkIndexes linkHidden(int layerDepth, int numOfNeurons);
    void linkOutput(void* (*method)(std::shared_ptr<NLayer> _upperLayer));
    void linkOutput(); // Default linker, all neurons from upper layer to every output
    void setLearningcurve(float curve, float min, float max);
    float getLearningrate(int depth);
    std::vector<std::shared_ptr<float> > getOutputSignals();
    std::vector<float> getSums();
    std::vector<std::vector<float> > getWeights(int depth);

    void forward();
    void back(std::vector<float> desiredOut);
    std::string getStats();
};