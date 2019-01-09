#include <math.h>
#include <iostream>
#include "Neuron.h"
#include "sigmoid.h"

Neuron::Neuron() : learningRate(0.1), sum(0), error(0) {
    outputSignal = std::make_shared<float>(0);
}

Neuron::Neuron(std::vector<std::shared_ptr<float> > ins_) : sum(0), error(0) {
    for (auto in_ : ins_) {
        addInput(in_);
    }
}

void Neuron::addInput(std::shared_ptr<float> out) {
    inputs.push_back(Input(out));
    inputs.back().setWeight(1);
    isInputNeuron = true;
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
    sum = 0;
    for (auto input : inputs) {
        sum += input.getWeightedInput();
    }
    if (!isInputNeuron){
        *outputSignal = sigmoid(sum);
    } else {
       *outputSignal = sum;
    }
}

void Neuron::back(float desiredOut) {
    error = -(desiredOut - *(outputSignal));
    float sigmoidDelta =  (*outputSignal) * (1 - (*outputSignal));
//    std::cout << "last out is: " << (*outputSignal) << "\n";
//    std::cout << "desired out is: " << desiredOut << "\n";
//    std::cout << "error is: " << error << "\n";

    for (int i = 1; i < inputs.size(); i++) {
            Input& input = inputs[i];
            float weight = input.getWeight();
            float correction = learningRate * error * sigmoidDelta * (*input.getInput());
//            std::cout << "weight is: " << weight << "\n";
//            std::cout << "correction is: " << correction << "\n";
            input.setWeight(weight - correction);
//            std::cout << "newWeight is: " << input.getWeight() << "\n";
        if (input.getNeuron()) {
            input.getNeuron()->error += error * sigmoidDelta * weight;
        }
    }
}

void Neuron::back() {
    float sigmoidDelta =  (*outputSignal) * (1 - (*outputSignal));
//    std::cout << "last out is: " << (*outputSignal) << "\n";
//    std::cout << "error is: " << error << "\n";
//    std::cout << "learningRate is: " << learningRate << "\n";
//    std::cout << "sigmoidDelta is: " << sigmoidDelta << "\n";
    for (int i = 1; i < inputs.size(); i++) {
        Input& input = inputs[i];
        float weight = input.getWeight();
        float errorDelta = learningRate * sigmoidDelta * error * (*input.getInput());
//        std::cout << "errorDelta is: " << errorDelta << "\n";
        input.setWeight(weight - errorDelta);
        if (input.getNeuron()) {
            input.getNeuron()->error += error * sigmoidDelta * weight;
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
