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

  printf("NeuralNet_create_layer:+%p count=%d\n", l, count);

  l->neurons = calloc(count, sizeof(Neuron));
  if (l->neurons == NULL) {
    status = STATUS_OOM;
    goto done;
  }
  l->count = count;
  status = STATUS_OK;

done:
  printf("NeuralNet_create_layer:-%p status=%d\n", l, StatusVal(status));
  return status;
}

static Status Neuron_init(Neuron* n, NeuronLayer* inputs) {
  Status status;
  double* weights;
  printf("Neuron_init:+%p inputs=%p\n", n, inputs);

  if (inputs == NULL) {
    weights = NULL;
  } else {
    // Calculate the initial weights. Note weights[0] is the bias
    // so we increase count of weights by one.
    int count = inputs->count + 1;
    weights = calloc(count, sizeof(double));
    if (weights == NULL) { status = STATUS_OOM; goto done; }

    for (int w = 0; w < count; w++) {
      weights[w] = rand0_1() - 0.5;
      printf("Neuron_init: %p weights[%d]=%lf\n", n, w, weights[w]);
    }
  }

  n->output = 0.0;
  n->weights = weights;
  n->inputs = inputs;
  status = STATUS_OK;

done:
  printf("Neuron_init:-%p status=%d\n", n, status);

  return status;
}

Status NeuralNet_init(NeuralNet* nn, int num_in_neurons, int num_hidden_layers,
    int num_out_neurons) {
  Status status;
  printf("NeuralNet_init:+%p num_in_neurons=%d num_hidden_layers=%d num_out_neurons=%d\n",
      nn, num_in_neurons, num_hidden_layers, num_out_neurons);

  nn->max_layers = num_hidden_layers + 2;
  nn->out_layer = nn->max_layers - 1;
  nn->last_hidden = 0;
  nn->layers = NULL;

  // Create the layers
  nn->layers = calloc(nn->max_layers, sizeof(NeuronLayer));
  if (nn->layers == NULL) { status = STATUS_OOM; goto done; }

  // Initalize input and output layers
  status = NeuralNet_create_layer(&nn->layers[0], num_in_neurons);
  if (StatusErr(status)) goto done;
  status = NeuralNet_create_layer(&nn->layers[nn->out_layer], num_in_neurons);
  if (StatusErr(status)) goto done;

  nn->layers[nn->max_layers-1].count = num_out_neurons;

  status = STATUS_OK;

done:
  if (StatusErr(status)) {
    NeuralNet_deinit(nn);
  }
  printf("NeuralNet_init:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

void NeuralNet_deinit(NeuralNet* nn) {
  Status status;
  printf("NeuralNet_deinit:+%p\n", nn);

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

  printf("NeuralNet_deinit:-%p\n", nn);
}

Status NeuralNet_add_hidden(NeuralNet* nn, int count) {
  Status status;

  printf("NeuralNet_add_hidden:+%p count=%d\n", nn, count);

  nn->last_hidden += 1;
  if (nn->last_hidden >= (nn->max_layers - 1)) {
    status = STATUS_TO_MANY_HIDDEN;
    goto done;
  }
  status = NeuralNet_create_layer(&nn->layers[nn->last_hidden], count);
  if (StatusErr(status)) goto done;
  status = STATUS_OK;

done:
  printf("NeuralNet_add_hidden:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

Status NeuralNet_start(NeuralNet* nn) {
  Status status;

  // Update out_layer if necessary
  if ((nn->last_hidden + 1) < (nn->max_layers - 1)) {
    // There were fewer hidden layers than expected so
    // move the output layer to be after the last hidden layer
    nn->out_layer = nn->last_hidden + 1;
    nn->layers[nn->out_layer].count = nn->layers[nn->max_layers - 1].count;
    nn->layers[nn->out_layer].neurons = nn->layers[nn->max_layers - 1].neurons;
    nn->layers[nn->max_layers - 1].count = 0;
    nn->layers[nn->max_layers - 1].neurons = NULL;
  }

  printf("NeuralNet_start:+%p max_layers=%d last_hidden=%d out_layer=%d\n",
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
    printf("NeuralNet_start: nn->layers[%d].count=%d\n", l, nn->layers[l].count);
    for (int n = 0; n < nn->layers[l].count; n++) {
      Neuron_init(&nn->layers[l].neurons[n], in_layer);
    }
  }

  status = STATUS_OK;

done:
  printf("NeuralNet_start:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

void NeuralNet_stop(NeuralNet* nn) {
  printf("NeuralNet_stop:+%p\n", nn);
  printf("NeuralNet_stop:-%p\n", nn);
}

void NeuralNet_inputs_(NeuralNet* nn, Pattern* input) {
  printf("NeuralNet_inputs_:+%p count=%d input_layer count=%d\n",
      nn, input->count, nn->layers[0].count);
  for (int n = 0; n < nn->layers[0].count; n++) {
    // "Calculate" the output for the input neurons
    Neuron* neuron = &nn->layers[0].neurons[n];
    neuron->output = input->data[n];
    printf("NeuralNet_inputs_: %p neuron=%p output=%lf\n", nn, neuron, neuron->output);
  }
  printf("NeuralNet_inputs_:-%p\n", nn);
}

void NeuralNet_process(NeuralNet* nn) {
  printf("NeuralNet_process_:+%p\n", nn);
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
      double weighted_sum = *weights++;

      // Loop thought all of the neuron's inputs summing it scaled by that inputs weight
      for (int i = 0; i < neuron->inputs->count; i++) {
        weighted_sum += weights[i] * inputs[i].output;
      }

      // Calcuate the output using a Sigmoidal Activation function
      neuron->output = 1.0/(1.0 + exp(-weighted_sum));
      printf("NeuralNet_process_: %p output=%lf weighted_sum=%lf\n",
          neuron, neuron->output, weighted_sum);
    }
  }
  printf("NeuralNet_process_:-%p\n", nn);
}

void NeuralNet_outputs_(NeuralNet* nn, Pattern* output) {
  int count;
  printf("NeuralNet_outputs_:+%p count=%d\n", nn, output->count);
  if (output->count > nn->layers[nn->out_layer].count) {
    count = nn->layers[nn->out_layer].count;
  } else {
    count = output->count;
  }
  for (int i = 0; i < count; i++) {
    output->data[i] = nn->layers[nn->out_layer].neurons[i].output;
    printf("NeuralNet_outputs_: %p output[%d]=%lf\n", nn, i, output->data[i]);
  }
  printf("NeuralNet_outputs_:-%p\n", nn);
}

double NeuralNet_adjust_(NeuralNet* nn, Pattern* output, Pattern* target) {
  printf("NeuralNet_adjust_:+%p output count=%d target count=%d\n",
      nn, output->count, target->count);

  // Calculate the network error and partial derivative of the error for the output layer
  nn->error = 0.0;
  assert(output->count == target->count);
  NeuronLayer* output_layer = &nn->layers[nn->out_layer];
  for (int i = 0; i < output->count; i++) {
    printf("NeuralNet_adjust_:-%p %d target=%lf output=%lf\n",
        nn, i, target->data[i], output->data[i]);

    // Compute the error as the difference between target and output
    double error = target->data[i] - output->data[i];

    // Compute the partial derivative of the activation w.r.t. error
    double pd_error = error * output->data[i] * (1.0 - output->data[i]);
    nn->layers[nn->out_layer].neurons[i].pd_error = pd_error;

    // Compute the sub of the square of the error and add to total_error
    double sse = 0.5 * error * error;
    nn->error += sse;
    printf("NeuralNet_adjust_:-%p %d diff_err=%lf  pd_error=%lf sse=%lf\n",
        nn, i, error, pd_error, sse);
  }
  printf("NeuralNet_adjust_: %p nn->error=%lf\n", nn, nn->error);

  // For all of layers starting at the output layer back propagate the pd_error
  // to the previous layers. The output layers pd_error has been calculated above
  int first_hidden_layer = 1;
  for (int l = nn->out_layer; l >= first_hidden_layer; l--) {
    NeuronLayer* cur_layer = &nn->layers[l];
    NeuronLayer* prev_layer = &nn->layers[l-1];
    printf("NeuralNet_adjust_: %p l=%d l-1=%d cur_layer=%p prev_layer=%p\n",
        nn, l, l-1, cur_layer, prev_layer);

    // Compute the partial derivative of the error for the previous layer
    for (int n = 0; n < prev_layer->count; n++) {
      double sum_weighted_pd_error = 0.0;
      for (int i = 0; i < cur_layer->count; i++) {
        double pd_error = cur_layer->neurons[i].pd_error;
        printf("NeuralNet_adjust_: %p cur_layer->neurons[%d]=%lf\n", nn, i, pd_error);
        double weight = cur_layer->neurons[i].weights[n];
        printf("NeuralNet_adjust_: %p cur_layer->neurons[%d].weights[%d]=%lf\n",
            nn, i, n, weight);
        sum_weighted_pd_error += pd_error * weight;
        printf("NeuralNet_adjust_: %p sum_weighted_pd_error=%lf\n", nn,
            sum_weighted_pd_error);
      }

      double prev_out = prev_layer->neurons[n].output;
      double pd_prev_out = prev_out * (1.0 - prev_out);
      printf("NeuralNet_adjust_: %p pd_prev_out:%lf = prev_out:%lf * (1.0 - prev_out:%lf)\n",
        nn, pd_prev_out, prev_out, prev_out);
      prev_layer->neurons[n].pd_error = sum_weighted_pd_error * pd_prev_out;
      printf("NeuralNet_adjust_: %p pd_error:%lf = sum_weighted_pd_error:%lf * pd_prev_out:%lf\n",
        nn, sum_weighted_pd_error, pd_prev_out);
    }
  }

#if 0
  // Update the weights for hidden and output
  for (int l = 1; l <= nn->out_layer; l--) {
    NeuronLayer* layer = &nn->layers[l];
    for (int n = 0; n < layer->count; n++) {
      Neuron* neuron = &layer->neurons[n];
      NeuronLayer* inputs = neuron->inputs;
      for (int i = 0; i < neuron->inputs->count; i++) {
        // Update the weights
      }
    }
  }
#endif

  printf("NeuralNet_adjust_:-%p nn->error=%lf\n", nn, nn->error);
  return nn->error;
}
