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
#define STATUS_OK 0
#define STATUS_ERR 1

/** Evaluates to true if status is good */
#define StatusOk(s) ((s) == STATUS_OK)

/** Evaluates to false if status is bad */
#define StatusErr(s) (!StatusOk(s))

/** Evaluates to the status integer value */
#define StatusVal(s) (s)

typedef struct {
  double* weights;
  int conn_num_in;
  int conn_num_out;
  double* conn_in;
  double* conn_out;
} Neuron;

typedef struct {
  int num_in;
  int num_out;
  int num_hidden;

  Neuron* neurons_in;
  Neuron* neurons_hidden;
  Neuron* neurons_out;
} NeuralNet;

Status Neuron_init(Neuron* n, int num_in, int num_out);
Status Neuron_init_conn_in(Neuron* n, int num_in, Neuron* neurons);
Status Neuron_init_conn_out(Neuron* n, int num_out, Neuron* neurons);

Status NeuralNet_init(NeuralNet* nn, int num_in, int num_out);
Status NeuralNet_deinit(NeuralNet* nn);
Status NeuralNet_add_hidden(NeuralNet* nn, int num_hidden, ...);

#endif
