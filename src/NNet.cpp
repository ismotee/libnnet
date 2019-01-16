#include <math.h>
#include <exception>

#include "NNet.h"
#include "NLayer.h" // for LayerLinksIndices
 
NNet::NNet() {

}

void NNet::createInputLayerAndLink(std::vector<std::shared_ptr<float> > inputs) {
    if (!layers.empty()) {
        std::cerr << "Warning: NNet layers not empty when calling createInputLayerAndLink().\n"
                << "This command will clear all layer architecture!\n";
    }
    layers.clear();
    layers.push_back(std::make_shared<InputLayer>());
    layers.back()->link(inputs);
}

void NNet::createHiddenLayerAndLink (int numOfNeurons) {
    if (layers.empty()) {
        // this needs a proper wrapping
        std::cerr << "Error: NNet layer empty when calling createHiddenLayerAndLink().\n"
                << "Please make input Layer first\n";
        throw std::exception();
    }
 
    std::shared_ptr<NLayer> upper = layers.back();
    layers.push_back(std::make_shared<HiddenLayer>());
    layers.back()->link(upper, numOfNeurons);
}

std::vector<std::shared_ptr<float> > NNet::createOutputLayerAndLink(int numOfOutputs) {
    if (layers.empty()) {
        // this needs a proper wrapping
        std::cerr << "Error: NNet layer empty when calling createOutputLayerAndLink().\n"
                << "Please make other layers first\n";
        throw std::exception();
    }

    std::shared_ptr<NLayer> upper = layers.back();
    layers.push_back(std::make_shared<OutputLayer>());
    layers.back()->link(upper, numOfOutputs);
    return layers.back()->getOutputSignals();
}

std::vector<std::shared_ptr<float> > NNet::getOutputSignals() {
    return layers.back()->getOutputSignals();
}


void NNet::setLearningcurve(float _curve, float _min, float _max) {
}

float NNet::getLearningrate(int depth) {
//    return std::pow(max - min, curve) * depth + max;
    return 0; 
}

void NNet::forward() {

    for(auto layer : layers) {
        layer->forward();
    }
}

void NNet::back(std::vector<float> desiredOut) {
    for(auto layer : layers) {
        layer->clearErrors();
    }
    for(int i = layers.size() - 1; i > 0; i--) {
        std::shared_ptr<NLayer> layer = layers[i];
        if (i == layers.size()-1) layer->back(desiredOut);
        layer->back();
    }
}

void NNet::printStats() {
    for(auto layer : layers) {
        std::cout << layer->getType() << ": " 
            << layer->getLayerSize() << "\n";
    }
}
