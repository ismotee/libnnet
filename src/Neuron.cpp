#include <math.h>
#include <iostream>
#include "Neuron.h"
#include "sigmoid.h"

class InputLayer;

Neuron::Neuron(NLayer& parentLayer) : sum(0), error(0), inhibitor(0) {
    outputSignal = std::make_shared<float>(0);
    layer = &parentLayer;
}

Neuron::Neuron(NLayer& parentLayer, std::vector<std::shared_ptr<float> > ins_) : sum(0), error(0), inhibitor(0) {
    for (auto in_ : ins_) {
        addInput(in_);
    }
    layer = &parentLayer;
}

void Neuron::addInput(std::shared_ptr<float> out) {
    inputs.push_back(Input(out));
    inputs.back().setWeight(1);
}

void Neuron::addInput(std::shared_ptr<float> out, float weight) {
    inputs.push_back(Input(out));
    inputs.back().setWeight(weight);
}

void Neuron::addInput(std::shared_ptr<Neuron> neuron) {
    inputs.push_back(Input(neuron));
}

void Neuron::addInputs(std::vector<std::shared_ptr<Neuron> > neurons_) {
    for (auto neuron : neurons_) {
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

    if (inhibitor > 0) {
        *outputSignal = -1;
        inhibitor--;
        return;
    }

    sum = 0;
    for (auto input : inputs) {
        if (*input.getInput() != -1) {
            sum += input.getWeightedInput();
        } 
    }
    if (layer->getType() != "Input Layer"){
        *outputSignal = sigmoid(sum);
    } else {
       *outputSignal = sum;
    }
    if (sum > layer->inhibitionTreshold && layer->inhibitionInterval != 0) {
        inhibitor = layer->inhibitionInterval;
    }
}

void Neuron::back(float desiredOut) {
    error = -(desiredOut - *(outputSignal));
    float sigmoidDelta =  (*outputSignal) * (1 - (*outputSignal));

    for (int i = 1; i < inputs.size(); i++) {
            Input& input = inputs[i];
            float weight = input.getWeight();
            float correction = layer->learningRate * error * sigmoidDelta * (*input.getInput());
            input.setWeight(weight - correction);
        if (input.getNeuron()) {
            input.getNeuron()->error += error * sigmoidDelta * weight;
        }
    }
}

void Neuron::back() {
    float sigmoidDelta =  (*outputSignal) * (1 - (*outputSignal));
    for (int i = 1; i < inputs.size(); i++) {
        Input& input = inputs[i];
        float weight = input.getWeight();
        float errorDelta = layer->learningRate * sigmoidDelta * error * (*input.getInput());
        input.setWeight(weight - errorDelta);
        if (input.getNeuron()) {
            input.getNeuron()->error += error * sigmoidDelta * weight;
        }
    }
}

float Neuron::getCurrentError() {
    return error;
}

std::vector<float> Neuron::getInputValues() {
    std::vector<float> result;
    for (auto in : inputs) {
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
