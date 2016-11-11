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

Status NeuralNet_init(NeuralNet* nn, int num_in, int num_out) {
  Status status;
  printf("NeuralNet_init:+%p num_in=%d num_out=%d\n", nn, num_in, num_out);

  nn->num_in = num_in;
  nn->num_out = num_out;
  status = STATUS_OK;

  printf("NeuralNet_init:-%p status=%d\n", StatusVal(status));
  return status;
}

Status NeuralNet_deinit(NeuralNet* nn) {
  Status status;
  printf("NeuralNet_deinit:+%p\n", nn);

  status = STATUS_OK;

  printf("NeuralNet_deinit:-%p status=%d\n", nn, StatusVal(status));
  return status;
}

Status NeuralNet_add_hidden(NeuralNet* nn, int num_hidden, ...) {
  Status status;
  va_list args;
  va_start(args, num_hidden);

  printf("NeuralNet_add_hidden:+%p num_hidden=%d\n", nn, num_hidden);

  for (int i = 0; i < num_hidden; i++) {
    printf("%p %d=%d\n", nn, i, va_arg(args, int));
  }
  status = STATUS_OK;

  printf("NeuralNet_add_hidden:-%p status=%d\n", nn, StatusVal(status));
  return status;
}
