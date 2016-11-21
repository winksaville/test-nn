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

#ifndef _NEURAL_NET_IO_H_
#define _NEURAL_NET_IO_H_

#include "NeuralNet.h"

#include <stdio.h>

typedef struct NeuralNetIoWriter NeuralNetIoWriter;

typedef void (*NeuralNetIoWriter_deinit)(NeuralNetIoWriter* writer);
typedef Status (*NeuralNetIoWriter_write_str)(NeuralNetIoWriter* writer, char* s);
typedef Status (*NeuralNetIoWriter_open_file)(NeuralNetIoWriter* writer, size_t epoch);
typedef Status (*NeuralNetIoWriter_close_file)(NeuralNetIoWriter* writer);
typedef Status (*NeuralNetIoWriter_begin_epoch)(NeuralNetIoWriter* writer, size_t epoch);
typedef Status (*NeuralNetIoWriter_end_epoch)(NeuralNetIoWriter* writer);

typedef struct NeuralNetIoWriter {
  FILE* out_file;       // Output file
  NeuralNet* nn;        // Neural net
  char* mode;           // Mode string
  char* out_dir;
  char* base_file_name;
  char* suffix;

  char file_name[1024];

  // Methods
  NeuralNetIoWriter_deinit deinit;
  NeuralNetIoWriter_open_file open_file;
  NeuralNetIoWriter_close_file close_file;
  NeuralNetIoWriter_begin_epoch begin_epoch;
  NeuralNetIoWriter_end_epoch end_epoch;
  NeuralNetIoWriter_write_str write_str;

} NeuralNetIoWriter;

Status NeuralNetIoWriter_init(NeuralNetIoWriter* writer, FILE* out_file, NeuralNet* nn, char* mode,
    char* out_dir, char* base_file_name, char* suffix);


#endif
