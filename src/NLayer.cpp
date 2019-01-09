#include <math.h>
#include <iostream>
#include "NLayer.h"

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

LayerLinkIndices methodAllCombinations(NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    LayerLinkIndices lLinks;
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

LayerLinkIndices methodEveryCombinationOfTwo(NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    LayerLinkIndices lLinks;
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

LayerLinkIndices methodOneSelectedAndOneRandom (NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    if(upperLayer->layer.size() <= 1) {
        std::cerr << "upper layer size must be over 1\n";
    }
    LayerLinkIndices result;
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



void NLayer::forward() {
    for (auto neuron : layer) {
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
    for(auto neuron: layer) {
	result.push_back(neuron->getWeights() );
    }
    return result;
}
std::vector<std::shared_ptr<Neuron> >* NLayer::getLayer() {
    return &layer;
}

void NLayer::setLearningRate(float lr) {
    for (auto neuron : layer) {
        neuron->setLearningRate(lr);
    }
}

int NLayer::getLayerSize() {
    return layer.size();
}

void NLayer::clearErrors() {
    for (auto neuron : layer) {
        neuron->error = 0;
    }
}

InputLayer::InputLayer() {
    bias = std::make_shared<float>(1);
    layer.push_back(std::make_shared<Neuron>());
    layer.back()->addInput(bias);
}

void InputLayer::link(std::shared_ptr<float> input) {
    layer.push_back(std::make_shared<Neuron>());
    layer.back()->addInput(input, 1);
}

void OutputLayer::back(std::vector<float> desiredOutputs) {
    if (desiredOutputs.size() != layer.size()) {
        std::cerr << "NLayer: mismatch of vector size: " << layer.size() << "/" << desiredOutputs.size() << "\n";
    }
    for (unsigned int i = 0; i < layer.size(); i++) {
        layer[i]->back(desiredOutputs[i]);
    }
}

HiddenLayer::HiddenLayer() {
    bias = std::make_shared<float>(1);
    layer.push_back(std::make_shared<Neuron>());
    layer.back()->addInput(bias);
}

HiddenLayer::HiddenLayer(int numOfInputs) {
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>());
    }
}

LayerLinkIndices HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndices(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)) {
    return method(this, upperLayer);
}

LayerLinkIndices HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, LinkMethod METHOD) {
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

LayerLinkIndices HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons) {
    LayerLinkIndices lLinks;
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

LayerLinkIndices OutputLayer::link(std::shared_ptr<NLayer> upperLayer, LayerLinkIndices(*method)(std::shared_ptr<NLayer>)) {
    return method(upperLayer);
}

LayerLinkIndices OutputLayer::link(std::shared_ptr<NLayer> upperLayer) {
    LayerLinkIndices result;
    std::vector<std::shared_ptr<Neuron> > linkableNeurons;
    for (unsigned int j = 0; j < layer.size(); j++) {
        linkableNeurons.clear();
        layer[j]->addInputs(upperLayer->layer);
    }
    return result;
}
