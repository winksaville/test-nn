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

NeuralNet nn;

int main(int argc, char** argv) {
  Status status;
  printf("test-nn:+\n");

  status = NeuralNet_init(&nn, 2, 1, 2);
  if (StatusErr(status)) goto done;

  status = NeuralNet_add_hidden(&nn, 4);
  if (StatusErr(status)) goto done;

done:
  NeuralNet_deinit(&nn);

  printf("test-nn:- status=%d\n", status);
  return 0;
}
