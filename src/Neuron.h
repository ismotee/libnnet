#pragma once
#include "Input.h"
#include <vector>
#include <memory>
#include "NLayer.h"

/**
 * @class Neuron
 * @author Ismo
 * @date 02.04.2018
 * @file libnnet.h
 * @brief Neuron with dynamic amount of inputs and one output. Neuron stores the brief history of Neuron activity.
 * Neuron takes care of inhibition as well as data that is needed for back propagation training of Input.
 */

class NLayer;

class Neuron {

    NLayer* layer;

    std::vector<Input> inputs;
    float sum;
    std::shared_ptr<float> outputSignal;
    int inhibitor;

public:
    float error;
    Neuron(NLayer& parentLayer);
    Neuron(NLayer& parentLayer, std::vector<std::shared_ptr<float> > ins_);
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
    std::vector<float> getInputValues();
    std::vector<float> getWeights();
};
