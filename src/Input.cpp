#include <iostream>

#include "Input.h"
#include "Neuron.h"

Input::Input() {
    weight = ((rand() % 100 - 30) * 0.001);
}

Input::Input(std::shared_ptr<float> in_) {
    weight = ((rand() % 100 - 30) * 0.001);
    in = in_;
}

Input::Input(std::shared_ptr<Neuron> neuron) {
    weight = ((rand() % 100 - 30) * 0.001);
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