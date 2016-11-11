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

#include <stdarg.h>
#include <stdio.h>
#include <malloc.h>

Status Neuron_init(Neuron* n, int num_in, int num_out) {
  Status status;
  printf("Neuron_init:+%p num_in=%d num_out=%d\n", n, num_in, num_out);

  n->conn_num_in = num_in;
  n->conn_num_out = num_out;
  status = STATUS_OK;

  printf("Neuron_init:-%p status=%d\n", n, StatusVal(status));
  return status;
}

Status Neuron_init_conn_in(Neuron* n, int num_in, Neuron* neurons) {
  Status status;
  printf("Neuron_init_conn_in:+%p num_in=%d num_in=%d\n", n, num_in);

  if (n->conn_num_in != num_in) {
    printf("Neuron_init_conn_in: %p error num_in:%d != conn_num_in=%d\n", n, num_in, n->conn_num_in);
    status = STATUS_ERR;
    goto done;
  }

  status = STATUS_OK;

done:
  printf("Neuron_init_conn_in:-%p status=%d\n", StatusVal(status));
  return status;
}

Status Neuron_init_conn_out(Neuron* n, int num_out, Neuron* neurons) {
  Status status;
  printf("Neuron_init_conn_out:+%p num_out=%d num_out=%d\n", n, num_out);

  if (n->conn_num_out != num_out) {
    printf("Neuron_init_conn_in: %p error num_out:%d != conn_num_out=%d\n", n, num_out, n->conn_num_out);
    status = STATUS_ERR;
    goto done;
  }

  status = STATUS_OK;

done:
  printf("Neuron_init_conn_out:-%p status=%d\n", StatusVal(status));
  return status;
}

Status NeuralNet_init(NeuralNet* nn, int num_in, int num_hidden, int num_out) {
  Status status;
  printf("NeuralNet_init:+%p num_in=%d num_hidden=%d num_out=%d\n", nn, num_in, num_hidden, num_out);

  nn->num_in = num_in;
  nn->num_hidden = num_hidden;
  nn->added_hidden = 0;
  nn->num_out = num_out;
  nn->neurons_in = NULL;
  nn->neurons_out = NULL;
  nn->neurons_hidden = NULL;

  nn->neurons_in = calloc(nn->num_in, sizeof(Neuron));
  if (nn->neurons_in == NULL) { status = STATUS_OOM; goto done; }

  nn->neurons_hidden = calloc(nn->num_hidden, sizeof(Neuron*));
  if (nn->neurons_hidden == NULL) { status = STATUS_OOM; goto done; }

  nn->neurons_out = calloc(nn->num_out, sizeof(Neuron));
  if (nn->neurons_out == NULL) { status = STATUS_OOM; goto done; }

  status = STATUS_OK;

done:
  if (StatusErr(status)) {
    NeuralNet_deinit(nn);
  }
  printf("NeuralNet_init:-%p status=%d\n", StatusVal(status));
  return status;
}

void NeuralNet_deinit(NeuralNet* nn) {
  Status status;
  printf("NeuralNet_deinit:+%p\n", nn);

  if (nn->neurons_in != NULL) {
    free(nn->neurons_in);
    nn->neurons_in = NULL;
  }

  if (nn->neurons_hidden != NULL) {
    for (int i = 0; i < nn->added_hidden; i++) {
      free(nn->neurons_hidden[i]);
      nn->neurons_hidden[i] = NULL;
    }
    free(nn->neurons_hidden);
    nn->neurons_hidden = NULL;
  }

  if (nn->neurons_out != NULL) {
    free(nn->neurons_out);
    nn->neurons_out = NULL;
  }

  printf("NeuralNet_deinit:-%p\n");
}

Status NeuralNet_add_hidden(NeuralNet* nn, int count) {
  Status status;

  printf("NeuralNet_add_hidden:+%p count=%d\n", nn, count);

  if (nn->added_hidden >= nn->num_hidden) {
    status = STATUS_TO_MANY_HIDDEN;
    goto done;
  }
  nn->neurons_hidden[nn->added_hidden] = calloc(count, sizeof(Neuron));
  if (nn->neurons_hidden[nn->added_hidden] == NULL) {
    status = STATUS_OOM;
    goto done;
  }
  status = STATUS_OK;

done:
  printf("NeuralNet_add_hidden:-%p status=%d\n", nn, StatusVal(status));
  return status;
}
