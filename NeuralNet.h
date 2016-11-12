/*
 * Copyright 2016 Wink Saville
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

typedef int Status;
#define STATUS_OK  0 //< OK
#define STATUS_ERR 1 //< Error
#define STATUS_OOM 2 //< Out of memory
#define STATUS_TO_MANY_HIDDEN 3 //< To many calls to NeuralNet_add_hidden

/** Evaluates to true if status is good */
#define StatusOk(s) ((s) == STATUS_OK)

/** Evaluates to false if status is bad */
#define StatusErr(s) (!StatusOk(s))

/** Evaluates to the status integer value */
#define StatusVal(s) (s)

typedef struct Neuron Neuron;
typedef struct NeuronLayer NeuronLayer;

typedef struct Pattern {
  int count;
  double data[];
} Pattern;

typedef struct Neuron {
  NeuronLayer* inputs;  // Neuron layer of inputs
  double* weights;      // Array of weights for each input plus the bias
  double output;        // The output of this neuron
} Neuron;

typedef struct NeuronLayer {
  int count;            // Number of neurons
  Neuron* neurons;      // The neurons
} NeuronLayer;

typedef struct NeuralNet {
  int max_layers;       // Maximum layers in the nn
                        // layers[0] input layer
                        // layers[1] first hidden layer
  int last_hidden;      // layers[last_hidden] is last hidden layer
  int out_layer;        // layers[out_layer] is output layer

  Pattern* input;       // Input pattern

  // There will always be at least two layers,
  // plus there are zero or more hidden layers.
  NeuronLayer* layers;
} NeuralNet;


Status NeuralNet_init(NeuralNet* nn, int num_in, int num_hidden, int num_out);
void NeuralNet_deinit(NeuralNet* nn);

Status NeuralNet_start(NeuralNet* nn);
void NeuralNet_stop(NeuralNet* nn);

Status NeuralNet_add_hidden(NeuralNet* nn, int count);

void NeuralNet_inputs_(NeuralNet* nn, Pattern* input);
#define NeuralNet_inputs(nn, i) NeuralNet_inputs_(nn, (Pattern*)i)

void NeuralNet_outputs_(NeuralNet* nn, Pattern* output);
#define NeuralNet_outputs(nn, o) NeuralNet_outputs_(nn, (Pattern*)o)

void NeuralNet_adjust_(NeuralNet* nn, Pattern* output, Pattern* target);
#define NeuralNet_adjust(nn, o, t) NeuralNet_adjust_(nn, (Pattern*)o, (Pattern*)t)

void NeuralNet_process(NeuralNet* nn);

#endif
