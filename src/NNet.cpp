#include <math.h>

#include "NNet.h"
#include "NLayer.h" // for LayerLinksIndices
 
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
//    inputLayer->setLearningRate(max);
}

LayerLinkIndices NNet::linkHidden(int layerDepth, LayerLinkIndices(*method)(NLayer* self, std::shared_ptr<NLayer>)) {
    LayerLinkIndices result;
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
//    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / (hiddenLayers.size() + 2)));
    return result;
}

LayerLinkIndices NNet::linkHidden(int layerDepth, LinkMethod METHOD) {
    LayerLinkIndices result;

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
//    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / hiddenLayers.size() + 2));
    return result;

}

LayerLinkIndices NNet::linkHidden(int layerDepth, int numOfNeurons) {
    LayerLinkIndices result;

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
//    hiddenLayers[0]->setLearningRate(getLearningrate(layerDepth / hiddenLayers.size() + 2));
    return result;


}

LayerLinkIndices NNet::linkOutput() {
    LayerLinkIndices result = outputLayer->link(hiddenLayers.back());
//    outputLayer->setLearningRate(min);
    return result;
}

LayerLinkIndices NNet::linkOutput(LayerLinkIndices (*method)(std::shared_ptr<NLayer> _upLayer)) {
    if (hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return LayerLinkIndices();
    }
    if (!outputLayer) {
        std::cerr << "outputLayer: no output layer found.\n";
        return LayerLinkIndices();
    }
    LayerLinkIndices result = outputLayer->link(hiddenLayers.back(), method);
    return result;
}

std::vector<std::shared_ptr<float> > NNet::getOutputSignals() {
    std::vector<std::shared_ptr<float> > result;
    for (auto neuron : (*outputLayer->getLayer())) {
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
    for (auto hLayer : hiddenLayers) {
        hLayer->forward();
    }
    outputLayer->forward();
}

void NNet::back(std::vector<float> desiredOut) {
    inputLayer->clearErrors();
    for (auto hid : hiddenLayers) {
        hid->clearErrors();
    }
    outputLayer->clearErrors();

    outputLayer->back(desiredOut);
    for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
        hiddenLayers[i]->back();
    }
//    inputLayer->back();

}

std::string NNet::getStats() {
    std::string result = "inputlayer: " + std::to_string(inputLayer->getLayer()->size()) + "\n";
    for (unsigned int i = 0; i < hiddenLayers.size(); i++) {
        result += "hiddenlayer" + std::to_string(i) + ": " + std::to_string(hiddenLayers[i]->getLayer()->size()) + "\n";
    }
    result += "outputlayer: " + std::to_string(outputLayer->getLayer()->size()) + "\n";
    return result;
}
