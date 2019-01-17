#include <math.h>
#include <iostream>
#include "NLayer.h"

void linkAll(NLayer* self, std::shared_ptr<NLayer> upperLayer) {
    for (unsigned int i = 0; i < self->layer.size(); i++) {
        self->layer[i]->addInputs(upperLayer->layer);
    }
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

void NLayer::back(std::vector<float> desiredOutputs) {
    if (desiredOutputs.size() != layer.size()) {
        std::cerr << "NLayer: mismatch of vector size: " << layer.size() << "/" << desiredOutputs.size() << "\n";
    }
    for (unsigned int i = 0; i < layer.size(); i++) {
        layer[i]->back(desiredOutputs[i]);
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
    learningRate = lr;
}

int NLayer::getLayerSize() {
    return layer.size();
}

void NLayer::clearErrors() {
    for (auto neuron : layer) {
        neuron->error = 0;
    }
}

void NLayer::link(std::shared_ptr<NLayer> upperLayer, void(*method)(NLayer* self, std::shared_ptr<NLayer> _upLayer)) {
    return method(this, upperLayer);
}

void NLayer::link(std::shared_ptr<NLayer> upperLayer, int numOfNeurons) {
    for (int i = 0; i < numOfNeurons; i++) {
        layer.push_back(std::make_shared<Neuron>(*this));
    }
    link(upperLayer, linkAll);
}

void NLayer::link(std::shared_ptr<float> input) {
    layer.push_back(std::make_shared<Neuron>(*this));
    layer.back()->addInput(input, 1);
}

void NLayer::link(std::vector<std::shared_ptr<float> > inputs) {
    for(auto input : inputs) {
        link(input);
    }
}

std::vector<std::shared_ptr<float> > NLayer::getOutputSignals() {
    std::vector<std::shared_ptr<float> > result;
    for (auto neuron : layer) {
        result.push_back(neuron->getOutputSignal());
    }
    return result;
}


InputLayer::InputLayer() {
    learningRate = 0.1;
    inhibitionInterval = 0;
    inhibitionTreshold = 0.7;
    bias = std::make_shared<float>(1);
    layer.push_back(std::make_shared<Neuron>(*this));
    layer.back()->addInput(bias);
}



HiddenLayer::HiddenLayer() {
    learningRate = 0.1;
    inhibitionInterval = 1;
    inhibitionTreshold = 0.7;
    bias = std::make_shared<float>(1);
    layer.push_back(std::make_shared<Neuron>(*this));
    layer.back()->addInput(bias);
}

HiddenLayer::HiddenLayer(int numOfInputs) {
    learningRate = 0.1;
    inhibitionInterval = 1;
    inhibitionTreshold = 0.7;
    bias = std::make_shared<float>(1);
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>(*this));
        layer.back()->addInput(bias);
    }
}


OutputLayer::OutputLayer() {
    learningRate = 0.1;
    inhibitionInterval = 0;
    inhibitionTreshold = 0.7;

}

OutputLayer::OutputLayer(int numOfInputs) {
    learningRate = 0.1;
    inhibitionInterval = 0;
    inhibitionTreshold = 0.7;
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>(*this));
    }
}