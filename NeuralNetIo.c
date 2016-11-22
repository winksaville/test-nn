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
#include "NeuralNetIo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static Status write_str(NeuralNetIoWriter* writer, char* data) {
  Status status;

  fputs(data, writer->out_file);
  if (ferror(writer->out_file)) {
     printf("NeuralNetIo.write: %s\n", strerror(errno));
  }

  status = STATUS_OK;
done:
  return status;
}

static void deinit(NeuralNetIoWriter* writer) {
  if (writer->mode != NULL) {
    free(writer->mode);
    writer->mode = NULL;
  }
}

static Status open_file(NeuralNetIoWriter* writer, size_t epoch) {
  Status status;

  snprintf(writer->file_name, sizeof(writer->file_name), "%s/%s%06d.%s",
      writer->out_dir, writer->base_file_name, epoch, writer->suffix);
  writer->out_file = fopen(writer->file_name, "w");
  if (writer->out_file == NULL) {
    printf("NeuralNetIo::start_epoch: could not open file: %s err=%s\n",
        writer->file_name, strerror(errno));
    status = STATUS_ERR;
    goto done;
  }

  // Write header
  char* header = "x y z value\n";
  status = writer->write_str(writer, header);
  if (StatusErr(status)) {
    printf("NeuralNetIoWriter_init: unable to write header\n");
    goto done;
  }

  status = STATUS_OK;

done:
  return status;
}

static Status close_file(NeuralNetIoWriter* writer) {
  Status status;

  fclose(writer->out_file);
  writer->out_file = NULL;

  status = STATUS_OK;

done:
  return status;
}


static Status begin_epoch(NeuralNetIoWriter* writer, size_t epoch) {
  Status status;

  status = writer->open_file(writer, epoch);
  if (StatusErr(status)) {
    goto done;
  }

  // Write header
  char* extra_data = "0.0 0.0 0.0\n1.0 1.0 1.0 0.0\n";
  status = writer->write_str(writer, extra_data);
  if (StatusErr(status)) {
    printf("NeuralNetIoWriter_init: unable to write extra_data\n");
    goto done;
  }

  status = STATUS_OK;

done:
  return status;
}

static Status write_epoch(NeuralNetIoWriter* writer) {
  Status status;

  NeuralNet* nn = writer->nn;
  double xaxis;
  double yaxis;
  double zaxis;

  double xaxis_max = 1.0;
  double yaxis_max = 1.0;
  double zaxis_max = 1.0;

  double xaxis_count = nn->out_layer + 1.0;
  xaxis_count += 1.0; // +1.0 for the output layer's output
  double xaxis_offset = xaxis_max / (xaxis_count + 1.0);

  double yaxis_count;
  double yaxis_offset;

  char buffer[1024];


  // Write the input neuron's output values
  yaxis_count = nn->layers[0].count;
  yaxis_offset = yaxis_max / (yaxis_count + 1.0);
  yaxis = yaxis_offset;

  xaxis = xaxis_offset;
  for (int n = 0; n < yaxis_count; n++) {
    Neuron* neuron = &nn->layers[0].neurons[n];
    snprintf(buffer, sizeof(buffer), "%lf %lf %lf %lf\n",
        xaxis, yaxis, neuron->output, neuron->output);
    writer->write_str(writer, buffer);
    yaxis += yaxis_offset;
  }

  // Write the hidden and output layers weights
  xaxis += xaxis_offset;
  for (int l = 1; l <= nn->out_layer; l++) {
    NeuronLayer* layer = &nn->layers[l];

    // Count the total connectons for the layer
    int yaxis_count = 0;
    for (int n = 0; n < layer->count; n++) {
      // The additional value is for the bias
      yaxis_count += (layer->neurons[0].inputs->count + 1);
    }
    yaxis_offset = yaxis_max / (yaxis_count + 1.0);

    yaxis = yaxis_offset;
    for (int n = 0; n < layer->count; n++) {
      // Get the next neuron
      Neuron* neuron = &layer->neurons[n];

      // Point at the first of the neuron's inputs and weights arrays
      double* weights = neuron->weights;

      // Loop thought all of the neuron's output the weights
      // which includes the bias, hence the <= test.
      for (int i = 0; i <= neuron->inputs->count; i++) {
        snprintf(buffer, sizeof(buffer), "%lf %lf %lf %lf\n",
            xaxis, yaxis, weights[i], weights[i]);
        writer->write_str(writer, buffer);
        yaxis += yaxis_offset;
      }
    }
    xaxis += xaxis_offset;
  }

  // Write the onput neuron's output values
  NeuronLayer* layer = &nn->layers[nn->out_layer];
  yaxis_count = layer->count;
  yaxis_offset = yaxis_max / (yaxis_count + 1.0);
  yaxis = yaxis_offset;

  for (int n = 0; n < yaxis_count; n++) {
    Neuron* neuron = &layer->neurons[n];
    snprintf(buffer, sizeof(buffer), "%lf %lf %lf %lf\n",
        xaxis, yaxis, neuron->output, neuron->output);
    writer->write_str(writer, buffer);
    yaxis += yaxis_offset;
  }

  status = STATUS_OK;

done:
  return status;
}

static Status end_epoch(NeuralNetIoWriter* writer) {
  Status status;

  writer->close_file(writer);

  status = STATUS_OK;

done:
  return status;
}

Status NeuralNetIoWriter_init(NeuralNetIoWriter* writer, FILE* out_file, NeuralNet* nn, char* mode,
    char* out_dir, char* base_file_name, char* suffix) {
  Status status;

  // Initialize
  writer->out_dir = out_dir;
  writer->base_file_name = base_file_name;
  writer->suffix = suffix;
  writer->out_file = out_file;
  writer->nn = nn;
  writer->mode = strdup(mode);
  writer->deinit = deinit;
  writer->open_file = open_file;
  writer->close_file = close_file;
  writer->begin_epoch = begin_epoch;
  writer->write_epoch = write_epoch;
  writer->end_epoch = end_epoch;
  writer->write_str = write_str;

  // Check
#if 0
  // Currently out_file is opened in begin_epoch
  if (writer->out_file == NULL) {
    printf("NeuralNetIoWriter_init: out_file is NULL\n");
    status = STATUS_BAD_PARAM;
    goto done;
  }
#endif
  if (writer->nn == NULL) {
    printf("NeuralNetIoWriter_init: nn is NULL\n");
    status = STATUS_BAD_PARAM;
  }
  if (writer->mode == NULL) {
    printf("NeuralNetIoWriter_init: mode is NULL\n");
    status = STATUS_BAD_PARAM;
  }

  status = STATUS_OK;

done:
  if (StatusErr(status)) {
    writer->deinit(writer);
  }

  return status;
}
