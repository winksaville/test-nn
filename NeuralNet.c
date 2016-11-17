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

#include "NeuralNet.h"
#include "dbg.h"

#include <malloc.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @return a double, N, such that 0.0 >= N < 1.0
 */
#define rand0_1() ((double)rand()/((double)RAND_MAX+1))

static Status NeuralNet_create_layer(NeuronLayer* l, int count) {
  Status status;

  dbg("NeuralNet_create_layer:+%p count=%d\n", l, count);

  l->neurons = calloc(count, sizeof(Neuron));
  if (l->neurons == NULL) {
    status = STATUS_OOM;
    goto done;
  }
  l->count = count;
  status = STATUS_OK;

done:
  dbg("NeuralNet_create_layer:-%p status=%d\n", l, StatusVal(status));
  return status;
}

static Status Neuron_init(Neuron* n, NeuronLayer* inputs) {
  Status status;
  double* weights;
  double* momentums;
  dbg("Neuron_init:+%p inputs=%p\n", n, inputs);

  if (inputs == NULL) {
    weights = NULL;
  } else {
    // Calculate the initial weights. Note weights[0] is the bias
    // so we increase count of weights by one.
    int count = inputs->count + 1;
    weights = calloc(count, sizeof(double));
    if (weights == NULL) { status = STATUS_OOM; goto done; }

    // Initialize weights >= -0.5 and < 0.5
    for (int w = 0; w < count; w++) {
      weights[w] = rand0_1() - 0.5;
      dbg("Neuron_init: %p weights[%d]=%lf\n", n, w, weights[w]);
    }

    // Allocate an array of mementums initialize to 0.0
    momentums = calloc(count, sizeof(double));
    if (momentums == NULL) { status = STATUS_OOM; goto done; }
  }

  n->inputs = inputs;
  n->weights = weights;
  n->momentums = momentums;
  n->output = 0.0;
  n->pd_error = 0.0;
  status = STATUS_OK;

done:
  dbg("Neuron_init:-%p status=%d\n", n, status);

  return status;
}

Status NeuralNet_init(NeuralNet* nn, int num_in_neurons, int num_hidden_layers,
    int num_out_neurons) {
  Status status;
  dbg("NeuralNet_init:+%p num_in_neurons=%d num_hidden_layers=%d num_out_neurons=%d\n",
      nn, num_in_neurons, num_hidden_layers, num_out_neurons);

  nn->max_layers = 2; // We always have an input and output layer
  nn->max_layers += num_hidden_layers; // Add num_hidden layers
  nn->out_layer = nn->max_layers - 1;  // last one is out_layer
  nn->last_hidden = 0; // No hidden layers yet
  nn->error = 0;       // No errors yet
  nn->learning_rate = 0.5; // Learning rate aka eta
  nn->momentum_factor = 0.9; // momemtum factor aka alpha
  nn->layers = NULL;   // No layers yet

  // Create the layers
  nn->layers = calloc(nn->max_layers, sizeof(NeuronLayer));
  if (nn->layers == NULL) { status = STATUS_OOM; goto done; }

  // Initalize input and output layers
  status = NeuralNet_create_layer(&nn->layers[0], num_in_neurons);
  if (StatusErr(status)) goto done;
  status = NeuralNet_create_layer(&nn->layers[nn->out_layer], num_out_neurons);
  if (StatusErr(status)) goto done;

  status = STATUS_OK;

done:
  if (StatusErr(status)) {
    NeuralNet_deinit(nn);
  }
  dbg("NeuralNet_init:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

void NeuralNet_deinit(NeuralNet* nn) {
  Status status;
  dbg("NeuralNet_deinit:+%p\n", nn);

  if (nn->layers != NULL) {
    for (int i = 0; i < nn->max_layers; i++) {
      if (nn->layers[i].neurons != NULL) {
        free(nn->layers[i].neurons);
      }
      nn->layers[i].neurons = NULL;
    }
    free(nn->layers);
    nn->max_layers = 0;
    nn->last_hidden = 0;
    nn->out_layer = 0;
    nn->layers = NULL;
  }

  dbg("NeuralNet_deinit:-%p\n", nn);
}

Status NeuralNet_add_hidden(NeuralNet* nn, int count) {
  Status status;

  dbg("NeuralNet_add_hidden:+%p count=%d\n", nn, count);

  nn->last_hidden += 1;
  if (nn->last_hidden >= (nn->max_layers - 1)) {
    status = STATUS_TO_MANY_HIDDEN;
    goto done;
  }
  status = NeuralNet_create_layer(&nn->layers[nn->last_hidden], count);
  if (StatusErr(status)) goto done;
  status = STATUS_OK;

done:
  dbg("NeuralNet_add_hidden:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

Status NeuralNet_start(NeuralNet* nn) {
  Status status;

  // Check if the user added all of the hidden layers they could
  if ((nn->last_hidden + 1) < (nn->max_layers - 1)) {
    // Nope, there were fewer hidden layers than there could be
    // so move the output layer to be after the last hidden layer
    nn->out_layer = nn->last_hidden + 1;
    nn->layers[nn->out_layer].count = nn->layers[nn->max_layers - 1].count;
    nn->layers[nn->out_layer].neurons = nn->layers[nn->max_layers - 1].neurons;
    nn->layers[nn->max_layers - 1].count = 0;
    nn->layers[nn->max_layers - 1].neurons = NULL;
  }

  dbg("NeuralNet_start:+%p max_layers=%d last_hidden=%d out_layer=%d\n",
      nn, nn->max_layers, nn->last_hidden, nn->out_layer);

  // Initialize the neurons for all of the layers
  for (int l = 0; l < nn->max_layers; l++) {
    NeuronLayer* in_layer;
    if (l == 0) {
      // Layer 0 is the input layer so it has no inputs
      in_layer == NULL;
    } else {
      in_layer = &nn->layers[l-1];
    }
    dbg("NeuralNet_start: nn->layers[%d].count=%d\n", l, nn->layers[l].count);
    for (int n = 0; n < nn->layers[l].count; n++) {
      Neuron_init(&nn->layers[l].neurons[n], in_layer);
    }
  }

  status = STATUS_OK;

done:
  dbg("NeuralNet_start:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

void NeuralNet_stop(NeuralNet* nn) {
  dbg("NeuralNet_stop:+%p\n", nn);
  dbg("NeuralNet_stop:-%p\n", nn);
}

void NeuralNet_inputs_(NeuralNet* nn, Pattern* input) {
  dbg("NeuralNet_inputs_:+%p count=%d input_layer count=%d\n",
      nn, input->count, nn->layers[0].count);
  for (int n = 0; n < nn->layers[0].count; n++) {
    // "Calculate" the output for the input neurons
    Neuron* neuron = &nn->layers[0].neurons[n];
    neuron->output = input->data[n];
    dbg("NeuralNet_inputs_: %p neuron=%p output=%lf\n", nn, neuron, neuron->output);
  }
  dbg("NeuralNet_inputs_:-%p\n", nn);
}

void NeuralNet_process(NeuralNet* nn) {
  dbg("NeuralNet_process_:+%p\n", nn);
  // Calcuate the output for the fully connected layers,
  // which start at nn->layers[1]
  for (int l = 1; l <= nn->out_layer; l++) {
    NeuronLayer* layer = &nn->layers[l];
    for (int n = 0; n < layer->count; n++) {
      // Get the next neuron
      Neuron* neuron = &layer->neurons[n];

      // Point at the first of the neuron's inputs and weights arrays
      Neuron* inputs = neuron->inputs->neurons;
      double* weights = neuron->weights;

      // Initialize the weighted_sum to the first weight, this is the bias
      double weighted_sum = *weights;

      // Skip past bias
      weights += 1;

      // Loop thought all of the neuron's inputs summing it scaled by that inputs weight
      for (int i = 0; i < neuron->inputs->count; i++) {
        weighted_sum += weights[i] * inputs[i].output;
      }

      // Calcuate the output using a Sigmoidal Activation function
      neuron->output = 1.0 / (1.0 + exp(-weighted_sum));
      dbg("NeuralNet_process_: %p output=%lf weighted_sum=%lf\n",
          neuron, neuron->output, weighted_sum);
    }
  }
  dbg("NeuralNet_process_:-%p\n", nn);
}

void NeuralNet_outputs_(NeuralNet* nn, Pattern* output) {
  int count;
  dbg("NeuralNet_outputs_:+%p count=%d\n", nn, output->count);
  if (output->count > nn->layers[nn->out_layer].count) {
    count = nn->layers[nn->out_layer].count;
  } else {
    count = output->count;
  }
  for (int i = 0; i < count; i++) {
    output->data[i] = nn->layers[nn->out_layer].neurons[i].output;
    dbg("NeuralNet_outputs_: %p output[%d]=%lf\n", nn, i, output->data[i]);
  }
  dbg("NeuralNet_outputs_:-%p\n", nn);
}

double NeuralNet_adjust_(NeuralNet* nn, Pattern* output, Pattern* target) {
  dbg("NeuralNet_adjust_:+%p output count=%d target count=%d\n",
      nn, output->count, target->count);

  // Calculate the network error and partial derivative of the error for the output layer
  dbg("\nNeuralNet_adjust_: %p calculate pd_error for output neurons and total_error\n", nn);
  nn->error = 0.0;
  assert(output->count == target->count);
  NeuronLayer* output_layer = &nn->layers[nn->out_layer];
  for (int n = 0; n < output->count; n++) {
    // Compute the error as the difference between target and output
    double err = target->data[n] - output->data[n];

    // Compute the partial derivative of the activation w.r.t. error
    double pd_err = err * output->data[n] * (1.0 - output->data[n]);
    nn->layers[nn->out_layer].neurons[n].pd_error = pd_err;

    // Compute the sub of the square of the error and add to total_error
    double sse = 0.5 * err * err;
    nn->error += sse;
    dbg("NeuralNet_adjust_: %p out_layer=%d:%d target=%lf output=%lf err=%lf pd_err=%lf sse=%lf\n",
        nn, nn->out_layer, n, target->data[n], output->data[n], err, pd_err, sse);
  }
  dbg("NeuralNet_adjust_: %p nn->error=%lf\n", nn, nn->error);

  // For all of layers starting at the output layer back propagate the pd_error
  // to the previous layers. The output layers pd_error has been calculated above
  dbg("\nNeuralNet_adjust_: %p backpropagate pd_error for the hidden layers\n", nn);
  int first_hidden_layer = 1;
  for (int l = nn->out_layer; l > first_hidden_layer; l--) {
    NeuronLayer* cur_layer = &nn->layers[l];
    NeuronLayer* prev_layer = &nn->layers[l-1];
    dbg("NeuralNet_adjust_: %p cur_layer=%d prev_layer=%d\n", nn, l, l-1);

    // Compute the partial derivative of the error for the previous layer
    for (int npl = 0; npl < prev_layer->count; npl++) {
      double sum_weighted_pd_err = 0.0;
      for (int ncl = 0; ncl < cur_layer->count; ncl++) {
        double pd_err = cur_layer->neurons[ncl].pd_error;
        dbg("NeuralNet_adjust_: %p cur_layer:%d:%d pd_err=%lf\n",
            nn, l, ncl, pd_err);
        double weight = cur_layer->neurons[ncl].weights[npl+1];
        dbg("NeuralNet_adjust_: %p cur_layer:%d:%d weights[%d]=%lf\n",
            nn, l, ncl, npl+1, weight);
        sum_weighted_pd_err += pd_err * weight;
        dbg("NeuralNet_adjust_: %p cur_layer:%d:%d sum_weighted_pd_err:%lf\n",
            nn, l, ncl, sum_weighted_pd_err);
      }

      double prev_out = prev_layer->neurons[npl].output;
      double pd_prev_out = prev_out * (1.0 - prev_out);
      dbg("NeuralNet_adjust_: %p prev_layer:%d:%d pd_prev_out:%lf = prev_out:%lf * (1.0 - prev_out:%lf)\n",
        nn, l-1, npl, pd_prev_out, prev_out, prev_out);
      prev_layer->neurons[npl].pd_error = sum_weighted_pd_err * pd_prev_out;
      dbg("NeuralNet_adjust_: %p prev_layer:%d:%d pd_error:%lf = sum_weighted_pd_err:%lf * pd_prev_out:%lf\n",
        nn, l-1, npl, prev_layer->neurons[npl].pd_error, sum_weighted_pd_err, pd_prev_out);
    }
  }

  // Update the weights for hidden layers and output layer
  dbg("\nNeuralNet_adjust_: %p update weights learning_rate=%lf momemutum_factor=%lf\n", nn, nn->learning_rate, nn->momentum_factor);
  for (int l = 1; l <= nn->out_layer; l++) {
    NeuronLayer* layer = &nn->layers[l];
    dbg("NeuralNet_adjust_: %p loop through layer %d\n", nn, l);
    for (int n = 0; n < layer->count; n++) {
      Neuron* neuron = &layer->neurons[n];
      NeuronLayer* inputs = neuron->inputs;

      // Point weights and mementums input entires at index 1,
      // the bias entries will be at index -1
      double* weights = &neuron->weights[1];
      double* momentums = &neuron->momentums[1];

      // Start with bias
      double pd_err = neuron->pd_error; 

      // Update the weights for bias
      double momentum = nn->momentum_factor * momentums[-1];
      dbg("momentum:%lf = nn->momentum_factor:%lf momentums[%d]:%lf\n",
          momentum, nn->momentum_factor, -1, momentums[-1]);
      momentums[-1] = (nn->learning_rate * pd_err) + momentum;
      dbg("NeuralNet_adjust_: %p %d:%d momentums[%d]:%lf = (eta:%lf * pd_err:%lf) + momentum:%lf bias\n",
          nn, l, n, -1, momentums[-1], nn->learning_rate, pd_err, momentum);

      double w = weights[-1];
      weights[-1] = weights[-1] + momentums[-1];
      dbg("NeuralNet_adjust_: %p %d:%d weights[%d]:%lf = weights[%d]:%lf momentums[%d]=%lf bias\n",
          nn, l, n, -1, weights[-1], -1, w, -1, momentums[-1]);


      // Loop through this neurons input neurons adjusting the weights and momentums
      dbg("NeuralNet_adjust_: %p loop through neurons for %d:%d update weights pd_err=%lf\n", nn, l, n, pd_err);
      for (int i = 0; i < neuron->inputs->count; i++) {
        // Update the weights
        double input = inputs->neurons[i].output;
        momentum = nn->momentum_factor * momentums[i];
        momentums[i]  = (nn->learning_rate * input * pd_err) + momentum;
        dbg("NeuralNet_adjust_: %p %d:%d momentums[%d]:%lf = (eta:%lf * input:%lf pd_err:%lf) + momentum:%lf\n",
            nn, l, n, i, momentums[i], nn->learning_rate, input, pd_err, momentum);

        w = weights[i];
        weights[i] = weights[i] + momentums[i];
        dbg("NeuralNet_adjust_: %p %d:%d weights[%d]:%lf = weights[%d]:%lf + momentums[%d]=%lf\n",
            nn, l, n, i, weights[i], i, w, i, momentums[i]);
      }
    }
  }

  dbg("NeuralNet_adjust_:-%p nn->error=%lf\n", nn, nn->error);
  return nn->error;
}
