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
    return_value = 1 / (1 + exp_value);

    return return_value;
}

bool checkBit (unsigned int value, unsigned char bit_n) {
    unsigned int bitMask = 1 << bit_n;
    if(value & bitMask) {
        return true;
    }
    return false;
}

std::vector<int> getBitNumbers (unsigned int value) {
    std::vector<int> result;
    
    for (unsigned int i = 0; i < 32; i++) {
        if(checkBit(value, i)) {
            result.push_back(i);
        }
    }
    return result;
}

Input::Input() {
    weight = rand() % 100000 * 0.00001;
}

Input::Input(std::shared_ptr<float> in_) {
    weight = rand() % 100000 * 0.00001;
    in = in_;
}

Input::Input(std::shared_ptr<Neuron> neuron) {
    weight = rand() % 100000 * 0.00001;
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

Neuron::Neuron() : learningRate(0.01), bias(1), error(0) {
    outputSignal = std::make_shared<float>(0);
}

Neuron::Neuron(std::vector<std::shared_ptr<float> > ins_) {
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

void Neuron::addInputs(std::vector<std::shared_ptr<Neuron> > ins_) {
    for (auto& in_ : ins_) {
        addInput(in_);
    }
    
}

std::shared_ptr<float> Neuron::getOutputSignal() {
    return outputSignal;
}

float Neuron::getOutput() {
    return output;
}

void Neuron::forward() {
    output = bias;
    for (auto& input : inputs) {
        output += input.getWeightedInput();
    }
    (*outputSignal) = sigmoid(output);
}

void Neuron::back(float desiredOut) {
    error =  (*outputSignal) - desiredOut;

    for (auto& input : inputs) {
        // sigmoidDelta is sig(in) - (1 - sig(in)) => last input == sig(in)
        float sigmoidDelta = (*input.getInput()) - (1 - (*input.getInput()));
        float weightDelta = -error * sigmoidDelta * (*input.getInput());
        input.setWeight(input.getWeight() + weightDelta * learningRate);
        input.getNeuron()->error += sigmoidDelta * error;
    }
}

void Neuron::back() {
    error /= inputs.size();
    for (auto& input : inputs) {
//        std::cout << error << "\n";
        // sigmoidDelta is sig(in) - (1 - sig(in)) => last input == sig(in)
        float sigmoidDelta = (*input.getInput()) - (1 - (*input.getInput()));
        float weightDelta = -error * sigmoidDelta * (*input.getInput());
        input.setWeight(input.getWeight() + weightDelta * learningRate);

        if(input.getNeuron()) {
            input.getNeuron()->error += sigmoidDelta * error;
        }
    }
}


float Neuron::getCurrentError() {
    return error;
}

void Neuron::setLearningRate(float lr) {
    learningRate = lr;
}

InputLayer::InputLayer() {

}

InputLayer::InputLayer(int numOfInputs) {
    for (int i = 0; i < numOfInputs; i++) {
        layer.push_back(std::make_shared<Neuron>());
    }
}

void InputLayer::link(std::shared_ptr<float> input) {
    layer.push_back(std::make_shared<Neuron>());
    layer.back()->addInput(input);
}


void OutputLayer::back(std::vector<float> desiredOutputs) {
    if (desiredOutputs.size() != layer.size()) {
        std::cerr << "NLayer: mismatch of vector size: " << layer.size() << "/" << desiredOutputs.size() << "\n";
    }
    for (int i = 0; i < layer.size(); i++) {
        layer[i]->back(desiredOutputs[i]);
    }
}

void NLayer::forward() {
    for (auto& neuron : layer) {
        neuron->forward();
    }
}

void NLayer::back() {
    for (int i = 0; i < layer.size(); i++) {
        layer[i]->back();
    }
}

std::vector<std::shared_ptr<Neuron> > NLayer::getLayer() {
    return layer;
}


void NLayer::setLearningRate(float lr) {
    for(auto& neuron : layer) {
        neuron->setLearningRate(lr);
    }
}

int NLayer::getLayerSize() {
    return layer.size();
}

void NLayer::clearErrors() {
    for(auto& neuron : layer) {
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

void HiddenLayer::link(std::shared_ptr<NLayer> upperLayer, void* (*method)(std::shared_ptr<NLayer> _upLayer)) {
    method(upperLayer);
}

void HiddenLayer::link(std::shared_ptr<NLayer> upperLayer) {
    layer.resize(std::pow(2, upperLayer->getLayerSize()) - 1, std::make_shared<Neuron>() );

    for (int i = 0; i < layer.size(); i++) {
        std::vector<int> links = getBitNumbers(i+1);
        
        for (int j = 0; j < links.size(); j++) {
            layer[i]->addInput(upperLayer->getLayer()[links[j]]);
        }
    }
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
    for(auto& neuron : layer) {
        neuron->addInputs(upperLayer->getLayer());
    }
}


NNet::NNet() {

}

NNet::NNet(int numOfInputs, int numOfOutputs, int numOfHiddenLayers) {    
    inputLayer = std::make_shared<InputLayer> (numOfInputs);
    outputLayer = std::make_shared<OutputLayer> (numOfOutputs);
    inputLayer->setLearningRate(max);
    outputLayer->setLearningRate(min);
    
    for (int i = 0; i < numOfHiddenLayers; i++) {
        hiddenLayers.push_back(std::make_shared<HiddenLayer>());
        hiddenLayers.back()->setLearningRate(getLearningrate(i+1));
    }
    outputLayer = std::make_shared<OutputLayer>(numOfOutputs);    
}

void NNet::linkInput(std::vector<std::shared_ptr<float> > input) {
    for(auto& in : input) {
        inputLayer->link(in);
    }
}


void NNet::linkHidden(int layerDepth, void*(*method)(std::shared_ptr<NLayer>)) {
    if(hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return;
    }
    if(!inputLayer) {
        std::cerr << "InputLayer: no input layer found.\n";
        return;
    }
    if(layerDepth > hiddenLayers.size()) {
        std::cerr << "HiddenLayer: no hiddenLayer with depth " << layerDepth << "\n";
        return;
    }
    
    if(layerDepth <= 1) {
        hiddenLayers[0]->link(inputLayer,method);
    } else {
        hiddenLayers[layerDepth-1]->link(hiddenLayers[layerDepth-2], method);
    }
}

void NNet::linkHidden(int layerDepth) {
    //TODO: tarkistukset
    if(layerDepth <= 1) {
        hiddenLayers[0]->link(inputLayer);    
    } else {
        hiddenLayers[layerDepth-1]->link(hiddenLayers[layerDepth-2]);
    }
}


void NNet::linkOutput() {
    outputLayer->link(hiddenLayers.back());
}


void NNet::linkOutput(void* (*method)(std::shared_ptr<NLayer> _upLayer)) {
    if(hiddenLayers.empty()) {
        std::cerr << "HiddenLayer: no hidden layers found.\n";
        return;
    }
    if(!outputLayer) {
        std::cerr << "outputLayer: no output layer found.\n";
        return;
    }
    outputLayer->link(hiddenLayers.back(), method);
}

std::vector<std::shared_ptr<float> > NNet::getOutputs() {
    std::vector<std::shared_ptr<float> > result;
    for(auto& neuron : outputLayer->getLayer()) {
        result.push_back(neuron->getOutputSignal());
    }
    return result;
}


void NNet::setLearningcurve(float _curve, float _min, float _max) {
    curve = _curve;
    min = _min;
    max = _max;
}

float NNet::getLearningrate(int depth) {
    return std::pow(max-min, curve) * depth + max;
}

void NNet::forward() {
    inputLayer->forward();
    for(auto& hLayer : hiddenLayers) {
        hLayer->forward();
    }
    outputLayer->forward();
}

void NNet::back(std::vector<float> desiredOut) {
    for (auto hid : hiddenLayers) {
        hid->clearErrors();
    }
    inputLayer->clearErrors();
    
    outputLayer->back(desiredOut);
    for (int i = hiddenLayers.size()-1; i >= 0; i--) {
        hiddenLayers[i]->back();
    }
    inputLayer->back();

}

