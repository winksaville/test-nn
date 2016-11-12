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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NeuralNet nn;

#define INPUT_COUNT 2
typedef struct InputPattern {
  int count;
  double data[INPUT_COUNT];
} InputPattern;

#define OUTPUT_COUNT 1
typedef struct OutputPattern {
  int count;
  double data[OUTPUT_COUNT];
} OutputPattern;

InputPattern or_input_patterns[] = {
  { .count = 2, .data[0] = 0, .data[1] = 0 },
  { .count = 2, .data[0] = 0, .data[1] = 1 },
  { .count = 2, .data[0] = 1, .data[1] = 0 },
  { .count = 2, .data[0] = 1, .data[1] = 1 },
};

OutputPattern or_target_patterns[] = {
  { .count = 1, .data[0] = 0 },
  { .count = 1, .data[0] = 1 },
  { .count = 1, .data[0] = 1 },
  { .count = 1, .data[0] = 1 },
};

int main(int argc, char** argv) {
  Status status;
  struct timespec spec;

  printf("test-nn:+\n");

  // seed the random number generator
#if 0
  clock_gettime(CLOCK_REALTIME, &spec);
  double dnow_us = (((double)spec.tv_sec * 1.0e9) + spec.tv_nsec) / 1.0e3;
  int now = (int)(long)dnow_us;
  printf("dnow_us=%lf now=0x%x\n", dnow_us, now);
  srand(now);
#else
  srand(0x12345678);
#endif

  status = NeuralNet_init(&nn, 2, 1, 1);
  if (StatusErr(status)) goto done;

  status = NeuralNet_add_hidden(&nn, 4);
  if (StatusErr(status)) goto done;

  status = NeuralNet_start(&nn);
  if (StatusErr(status)) goto done;


  OutputPattern or_output;

  NeuralNet_inputs(&nn, &or_input_patterns[0]);
  NeuralNet_process(&nn);
  NeuralNet_outputs(&nn, &or_output);
  NeuralNet_adjust(&nn, &or_output, &or_target_patterns[0]);

  double error = 0.0;
  for (int i = 0; i < or_output.count; i++) {
    double diff = or_target_patterns[0].data[i] - or_output.data[i];
    printf("diff=%lf\n", diff);
    error += 0.5 * (diff * diff);
  }
  printf("Total error=%lf\n", error);

  NeuralNet_stop(&nn);

done:
  NeuralNet_deinit(&nn);

  printf("test-nn:- status=%d\n", status);
  return 0;
}
