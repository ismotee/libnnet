#include "libnnet.h"
#include <random>
#include <algorithm>
#include <math.h>
#include <iostream>

/**********************************************************************/
/*** sigmoid.c:  This code contains the function routine            ***/
/***             sigmoid() which performs the unipolar sigmoid      ***/
/***             function for backpropagation neural computation.   ***/
/***             Accepts the input value x then returns it's        ***/
/***             sigmoid value in float.                            ***/
/***                                                                ***/
/***  function usage:                                               ***/
/***       float sigmoid(float x);                                  ***/
/***           x:  Input value                                      ***/
/***                                                                ***/
/***  Written by:  Kiyoshi Kawaguchi                                ***/
/***               Electrical and Computer Engineering              ***/
/***               University of Texas at El Paso                   ***/
/***  Last update:  09/28/99  for version 2.0 of BP-XOR program     ***/

/**********************************************************************/


float sigmoid(float x) {
    float exp_value;
    float return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double) -x);

    /*** Final sigmoid value ***/
    return_value = (float) 1 / (1 + exp_value);

    return return_value;
}

bool checkBit(unsigned int value, unsigned char bit_n) {
    unsigned int bitMask = 1 << bit_n;
    if (value & bitMask) {
        return true;
    }
    return false;
}

std::vector<int> getBitNumbers(unsigned int value) {
    std::vector<int> result;

    for (unsigned int i = 0; i < 32; i++) {
        if (checkBit(value, i)) {
            result.push_back(i);
        }
    }
    return result;
}

LayerLinkIndexes methodAllCombinations(NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    LayerLinkIndexes lLinks;
    int layerSize = std::pow(2, upperLayer->getLayerSize()) - 1;
    for (int i = 0; i < layerSize; i++) {
        self->layer.push_back(std::make_shared<Neuron>());
    }
    for (int i = 0; i < self->getLayerSize(); i++) {
        std::vector<int> links = getBitNumbers(i + 1);
        std::vector<std::shared_ptr<Neuron> > neurons;
        for (unsigned int j = 0; j < links.size(); j++) {
            neurons.push_back(upperLayer->layer[links[j]]);
        }
        self->layer[i]->addInputs(neurons);
        lLinks.push_back(links);
    }
    return lLinks;
}

LayerLinkIndexes methodEveryCombinationOfTwo(NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    LayerLinkIndexes lLinks;
    int layerSize = (float) upperLayer->getLayerSize() * (upperLayer->getLayerSize() + 1) / 2;
    for (int i = 0; i < layerSize; i++) {
        self->getLayer()->push_back(std::make_shared<Neuron>());
    }
    for (int i = 0; i < self->getLayerSize(); i++) {
        std::vector<int> links = getBitNumbers(i + 1);
        if (links.size() <= 2) {
            std::vector<std::shared_ptr<Neuron> > neurons;
            for (unsigned int j = 0; j < links.size(); j++) {
                neurons.push_back(upperLayer->getLayer()->at(links[j]));
            }
            self->getLayer()->at(i)->addInputs(neurons);
            lLinks.push_back(links);
        }
    }
    return lLinks;
}

LayerLinkIndexes methodOneSelectedAndOneRandom (NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    if(upperLayer->layer.size() <= 1) {
        std::cerr << "upper layer size must be over 1\n";
    }
    LayerLinkIndexes result;
    int randomIdx;
    for(int i = 0; i < upperLayer->layer.size(); i++) {
        do {
            randomIdx = std::rand() % upperLayer->layer.size();
        } while(randomIdx == i);
        std::vector<std::shared_ptr<Neuron> > neurons = {
            upperLayer->layer[i],
            upperLayer->layer[randomIdx]
        };
        self->layer.push_back(std::make_shared<Neuron>());
        self->layer.back()->addInputs(neurons);
        
        std::vector<int> links = {i, randomIdx};
        result.push_back(links);
        
    }
    return result;
}

Input::Input() {
    weight = (rand() % 100 * 0.01);
    std::cout << weight << "\n";
}

Input::Input(std::shared_ptr<float> in_) {
    weight = (rand() % 100 * 0.01);
    std::cout << weight << "\n";
    in = in_;
}

Input::Input(std::shared_ptr<Neuron> neuron) {
    weight = (rand() % 100 * 0.01);
    std::cout << weight << "\n";
    linkedNeuron = neuron;
    in = linkedNeuron->getOutputSignal();
}

void Input::connect(std::shared_ptr<float> in_) {
    in = in_;
}

void Input::connect(std::shared_ptr<Neuron> neuron) {
    linkedNeuron = neuron;
    in = linkedNeuron->getOutputSignal();
}

void Input::disconnect() {
    in = nullptr;
}

float Input::getWeight() {
    return weight;
}

float Input::getWeightedInput() {
    return weight * (*in);
}

void Input::setWeight(float w) {
    weight = w;
}

std::shared_ptr<float> Input::getInput() {
    return in;
}

std::shared_ptr<Neuron> Input::getNeuron() {
    return linkedNeuron;
}

Neuron::Neuron(float bias) : learningRate(0.01), sum(0), error(0) {
    addInput(std::make_shared<float>(bias));
    outputSignal = std::make_shared<float>(0);
}

Neuron::Neuron(std::vector<std::shared_ptr<float> > ins_, float bias) : sum(0), error(0) {
    addInput(std::make_shared<float>(bias));
    for (auto& in_ : ins_) {
        addInput(in_);
    }
}

void Neuron::addInput(std::shared_ptr<float> out) {
    inputs.push_back(Input(out));
}

void Neuron::addInput(std::shared_ptr<Neuron> neuron) {
    inputs.push_back(Input(neuron));
}

void Neuron::addInputs(std::vector<std::shared_ptr<Neuron> > neurons_) {
    for (auto& neuron : neurons_) {
        addInput(neuron);
    }

}

std::shared_ptr<float> Neuron::getOutputSignal() {
    return outputSignal;
}

float Neuron::getSum() {
    return sum;
}

void Neuron::forward() {
    sum = 0;
    for (auto& input : inputs) {
        sum += input.getWeightedInput();
    }
    *outputSignal = sigmoid(sum);
}

void Neuron::back(float desiredOut) {
    error = desiredOut - *outputSignal;
    float sigmoidDelta = error * (*outputSignal) * (1 - (*outputSignal));
    for (auto& input : inputs) {
        if (input.getNeuron()) {
            float weight = input.getWeight();
            input.setWeight(weight + learningRate * sigmoidDelta * (*input.getInput()));
            input.getNeuron()->error += sigmoidDelta * input.getWeight();
        }
    }
}

void Neuron::back() {
    float sigmoidDelta = (*outputSignal)* (1 - (*outputSignal));
    for (auto& input : inputs) {
            float weight = input.getWeight();
            float errorDelta = learningRate * sigmoidDelta * error * (*input.getInput());
            input.setWeight(weight + errorDelta);

            // add error for next layer except if layer is input layer of input is bias
        if (input.getNeuron()) {
            input.getNeuron()->error += errorDelta * input.getWeight();
        }
    }
}

float Neuron::getCurrentError() {
    return error;
}

void Neuron::setLearningRate(float lr) {
    learningRate = lr;
}

std::vector<float> Neuron::getInputValues() {
    std::vector<float> result;
    for (auto& in : inputs) {
        result.push_back(*in.getInput());
    }
    return result;
}

std::vector<float> Neuron::getWeights() {
    std::vector<float> result;
    for(unsigned int i=0; i<inputs.size(); i++) {
        result.push_back(inputs[i].getWeight() );
    }
    return result;
}

InputLayer::InputLayer() {

}

void InputLayer::link(std::shared_ptr<float> input) {
    layer.push_back(std::make_shared<Neuron>(0));
    layer.back()->addInput(input);
}

void OutputLayer::back(std::vector<float> desiredOutputs) {
    if (desiredOutputs.size() != layer.size()) {
        std::cerr << "NLayer: mismatch of vector size: " << layer.size() << "/" << desiredOutputs.size() << "\n";
    }
    for (unsigned int i = 0; i < layer.size(); i++) {
        layer[i]->back(desiredOutputs[i]);
    }
}

void NLayer::forward() {
    for (auto& neuron : layer) {
        neuron->forward();
    }
}

void NLayer::back() {
    for (unsigned int i = 0; i < layer.size(); i++) {
        layer[i]->back();
    }
}

std::vector<std::vector<float> > NLayer::getWeights() {
    std::vector<std::vector<float> > result;
    for(auto& neuron: layer) {
	result.push_back(neuron->getWeights() );
    }
    return result;
}
std::vector<std::shared_ptr<Neuron> >* NLayer::getLayer() {
    return &layer;
}

void NLayer::setLearningRate(float lr) {
    for (auto& neuron : layer) {
        neuron->setLearningRate(lr);
    }
}

int NLayer::getLayerSize() {
    return layer.size();
}

void NLayer::clearErrors() {
    for (auto& neuron : layer) {
        neuron->error = 0;
    }
}

HiddenLayer::HiddenLayer() {

}

HiddenLayer::HiddenLayer(int numOfInputs) {
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>());
    }
}

LayerLinkIndexes HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndexes(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)) {
    return method(this, upperLayer);
}

LayerLinkIndexes HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, LinkMethod METHOD) {
    switch (METHOD) {
        case ALL_COMBINATIONS:
            return link(upperLayer, methodAllCombinations);
        case EVERY_COMBINATION_OF_TWO:
            return link(upperLayer, methodEveryCombinationOfTwo);
        case ONE_SELECTED_AND_ONE_RANDOM:
            return link(upperLayer, methodOneSelectedAndOneRandom);
        default:
            return std::vector<std::vector<int> >();
    }
}

LayerLinkIndexes HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons) {
    LayerLinkIndexes lLinks;
    for (int i = 0; i < numOfNeurons; i++) {
        layer.push_back(std::make_shared<Neuron>());
    }
    for (int i = 0; i < layer.size(); i++) {
        std::vector<int> links;
        std::vector<std::shared_ptr<Neuron> > neurons;
        for (unsigned int j = 0; j < upperLayer->layer.size(); j++) {
            neurons.push_back(upperLayer->layer[j]);
            links.push_back(j);
        }
        layer[i]->addInputs(neurons);
        lLinks.push_back(links);
    }
    return lLinks;
}

OutputLayer::OutputLayer() {

}

OutputLayer::OutputLayer(int numOfInputs) {
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>());
    }
}

void OutputLayer::link(std::shared_ptr<NLayer> upperLayer, void*(*method)(std::shared_ptr<NLayer>)) {
    method(upperLayer);
}

void OutputLayer::link(std::shared_ptr<NLayer> upperLayer) {
    std::vector<std::shared_ptr<Neuron> > linkableNeurons;
    for (unsigned int j = 0; j < layer.size(); j++) {
        linkableNeurons.clear();
        for (unsigned int i = 0; i < upperLayer->layer.size(); i++) {
            if ((int) i % layer.size() == j) {
                linkableNeurons.push_back(upperLayer->layer[i]);
            }
        }
        layer[j]->addInputs(linkableNeurons);
    }
}

NNet::NNet() {

}

NNet::NNet(int numOfOutputs, int numOfHiddenLayers) {
    inputLayer = std::make_shared<InputLayer> ();
    outputLayer = std::make_shared<OutputLayer> (numOfOutputs);

    for (int i = 0; i < numOfHiddenLayers; i++) {
        hiddenLayers.push_back(std::make_shared<HiddenLayer>());
    }
    outputLayer = std::make_shared<OutputLayer>(numOfOutputs);
}

void NNet::linkInput(std::vector<std::shared_ptr<float> > input) {
    for (auto& in : input) {
        inputLayer->link(in);
    }
    inputLayer->setLearningRate(max);
}

LayerLinkIndexes NNet::linkHidden(int layerDepth, LayerLinkIndexes(*method)(NLayer* self, std::shared_ptr<NLayer>)) {
    LayerLinkIndexes result;
    if (hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return result;
    }
    if (!inputLayer) {
        std::cerr << "InputLayer: no input layer found.\n";
        return result;
    }
    if (layerDepth > (int) hiddenLayers.size()) {
        std::cerr << "HiddenLayer: no hiddenLayer with depth " << layerDepth << "\n";
        return result;
    }
    if (layerDepth <= 1) {
        result = hiddenLayers[0]->link(inputLayer, method);
    } else {
        result = hiddenLayers[layerDepth - 1]->link(hiddenLayers[layerDepth - 2], method);
    }
    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / (hiddenLayers.size() + 2)));
    return result;
}

LayerLinkIndexes NNet::linkHidden(int layerDepth, LinkMethod METHOD) {
    LayerLinkIndexes result;

    if (hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return result;
    }
    if (!inputLayer) {
        std::cerr << "InputLayer: no input layer found.\n";
        return result;
    }
    if (layerDepth > (int) hiddenLayers.size()) {
        std::cerr << "HiddenLayer: no hiddenLayer with depth " << layerDepth << "\n";
        return result;
    }

    if (layerDepth <= 1) {
        result = hiddenLayers[0]->link(inputLayer, METHOD);
    } else {
        result = hiddenLayers[layerDepth - 1]->link(hiddenLayers[layerDepth - 2], METHOD);
    }
    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / hiddenLayers.size() + 2));
    return result;

}

LayerLinkIndexes NNet::linkHidden(int layerDepth, int numOfNeurons) {
    LayerLinkIndexes result;

    if (hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return result;
    }
    if (!inputLayer) {
        std::cerr << "InputLayer: no input layer found.\n";
        return result;
    }
    if (layerDepth > (int) hiddenLayers.size()) {
        std::cerr << "HiddenLayer: no hiddenLayer with depth " << layerDepth << "\n";
        return result;
    }

    if (layerDepth <= 1) {
        result = hiddenLayers[0]->link(inputLayer, numOfNeurons);
    } else {
        result = hiddenLayers[layerDepth - 1]->link(hiddenLayers[layerDepth - 2], numOfNeurons);
    }
    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / hiddenLayers.size() + 2));
    return result;


}

void NNet::linkOutput() {
    outputLayer->link(hiddenLayers.back());
    outputLayer->setLearningRate(min);
}

void NNet::linkOutput(void* (*method)(std::shared_ptr<NLayer> _upLayer)) {
    if (hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return;
    }
    if (!outputLayer) {
        std::cerr << "outputLayer: no output layer found.\n";
        return;
    }
    outputLayer->link(hiddenLayers.back(), method);
}

std::vector<std::shared_ptr<float> > NNet::getOutputSignals() {
    std::vector<std::shared_ptr<float> > result;
    for (auto& neuron : (*outputLayer->getLayer())) {
        result.push_back(neuron->getOutputSignal());
    }
    return result;
}

std::vector<float> NNet::getSums() {
    std::vector<float> result;
    for (auto& neuron : (*outputLayer->getLayer())) {
        result.push_back(neuron->getSum());
    }
    return result;
}

std::vector<std::vector<float> > NNet::getWeights (int depth) {
    if(depth <= 0) {
       return inputLayer->getWeights();
    } else if (depth <= hiddenLayers.size()) {
       return hiddenLayers[depth - 1]->getWeights();
    } else {
       return outputLayer->getWeights();
    }
}

void NNet::setLearningcurve(float _curve, float _min, float _max) {
    curve = _curve;
    min = _min;
    max = _max;
}

float NNet::getLearningrate(int depth) {
    return std::pow(max - min, curve) * depth + max;
}

void NNet::forward() {
    inputLayer->forward();
    for (auto& hLayer : hiddenLayers) {
        hLayer->forward();
    }
    outputLayer->forward();
}

void NNet::back(std::vector<float> desiredOut) {
    inputLayer->clearErrors();
    for (auto& hid : hiddenLayers) {
        hid->clearErrors();
    }
    outputLayer->clearErrors();

    outputLayer->back(desiredOut);
    for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
        hiddenLayers[i]->back();
    }
    inputLayer->back();

}

std::string NNet::getStats() {
    std::string result = "inputlayer: " + std::to_string(inputLayer->getLayer()->size()) + "\n";
    for (unsigned int i = 0; i < hiddenLayers.size(); i++) {
        result += "hiddenlayer" + std::to_string(i) + ": " + std::to_string(hiddenLayers[i]->getLayer()->size()) + "\n";
    }
    result += "outputlayer: " + std::to_string(outputLayer->getLayer()->size()) + "\n";
    return result;
}
