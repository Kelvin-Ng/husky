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

#include <cmath>
#include <vector>

#include "core/engine.hpp"

namespace husky {
namespace lib {

template <typename T, bool is_sparse>
class Vector;

template <typename T>
class VectorBase {
   public:
    VectorBase<T> operator-() const;

    VectorBase<T> operator*(T) const;
    VectorBase<T>& operator*=(T);
    VectorBase<T> scalar_multiple_with_intcpt(T) const;

    VectorBase<T> operator/(T) const;
    VectorBase<T>& operator/=(T);
};

template <typename T>
class SequencialAccessVectorBase : public VectorBase<T> {

};

template <typename T>
class RandomAccessVectorBase : public SequencialAccessVectorBase<T> {
};

template <typename T>
class Vector<T, false> : public RandomAccessVectorBase<T> {
};

template <typename T>
class Vector<T, true> : public SequencialAccessVectorBase<T> {
};

}  // namespace lib
}  // namespace husky
