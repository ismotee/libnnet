#pragma once
#include "Input.h"
#include <vector>
#include <memory>

/**
 * @class Neuron
 * @author Ismo
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Neuron with dynamic amount of inputs and one output. Neuron stores the brief history of Neuron activity.
 * Neuron takes care of inhibition as well as data that is needed for back propagation training of Input.
 */

class Neuron {
    bool isInputNeuron;
    float learningRate;
    std::vector<Input> inputs;
    float sum;
    std::shared_ptr<float> outputSignal;

public:
    float error;
    Neuron();
    Neuron(std::vector<std::shared_ptr<float> > ins_);
    void addInput(std::shared_ptr<float> out);
    void addInput(std::shared_ptr<float> out, float weight);
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
