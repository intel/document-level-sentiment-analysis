#!/bin/bash

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

#

GCC_GOOD=`gcc --version | awk '/gcc/ && ($3+0)>=8.3{print "1"}'`
if [ "x$GCC_GOOD" != "x1" ] ; then
  echo "Requires gcc version later than 8.3.0"
  exit 1
fi

pt_version=$(python -c "import torch; print(torch.__version__)" 2> /dev/null)
if [ "x$pt_version" == "x" ] ; then
  echo "Can't find pytorch version, need PyTorch 1.9 or higher..."
  exit 1
fi

branch=$(echo $pt_version | tr "." " " | awk '{print "ccl_torch" $1 "." $2}')

if ! test -d ./torch-ccl ; then
  git clone https://github.com/intel/torch-ccl.git
fi
cd torch-ccl
# workaround to disable linker error for linking to mkl libraries
# export CMAKE_FIND_DEBUG_MODE=ON
export CMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
git checkout $branch && git submodule sync && git submodule update --init --recursive && CC=gcc CXX=g++ CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ python setup.py install

