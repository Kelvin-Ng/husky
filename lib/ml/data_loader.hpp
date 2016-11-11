// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/input/line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"

namespace husky {
namespace lib {
namespace ml {
typedef std::vector<double> vec_double;
typedef std::vector<std::pair<int, double>> vec_sp;

// indicate format
const int kLIBSVMFormat = 0;
const int kTSVFormat = 1;

template <typename ObjT = FeatureLabel>
class DataLoader {
    typedef ObjList<ObjT> ObjL;

   public:
    DataLoader() {}
    explicit DataLoader(int _format);
    explicit DataLoader(std::function<void(std::string, ObjL&)> _load_func) { load_func_ = _load_func; }

    void load_info(std::string url, ObjL& data) {
        ASSERT_MSG(load_func_ != nullptr, "Load function is not specified.");
        load_func_(url, data);
    }

    inline int get_num_feature() const { return this->num_feature_; }

   protected:
    int num_feature_ = -1;
    int format_;
    std::function<void(std::string, ObjL&)> load_func_ = nullptr;
};

template <>
DataLoader<FeatureLabel>::DataLoader(int _format);

template <>
DataLoader<SparseFeatureLabel>::DataLoader(int _format);

template <typename ObjT>
DataLoader<ObjT>::DataLoader(int _format) {
    this->format_ = _format;
}

}  // namespace ml
}  // namespace lib
}  // namespace husky
