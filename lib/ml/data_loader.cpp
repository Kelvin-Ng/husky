#include "data_loader.hpp"

namespace husky {
namespace lib {
namespace ml {

template <>
DataLoader<FeatureLabel>::DataLoader(int _format)
    : DataLoader([this](std::string url, DataLoader<FeatureLabel>::ObjL& data) {

          husky::io::LineInputFormat infmt;
          infmt.set_input(url);

          husky::lib::Aggregator<int> num_features_agg(0, [](int& a, const int& b) { a = b; });
          auto& ac = husky::lib::AggregatorFactory::get_channel();

          std::function<void(boost::string_ref)> parser;
          if (this->format_ == kLIBSVMFormat) {
              // LIBSVM format -- [label] [featureIndex]:[featureValue] [featureIndex]:[featureValue]
              parser = [&](boost::string_ref chunk) {
                  if (chunk.empty())
                      return;
                  boost::char_separator<char> sep(" \t");
                  boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                  FeatureLabel this_obj;
                  double& label = this_obj.use_label();
                  vec_double& feature = this_obj.use_feature();

                  int i = 0;
                  for (auto& w : tok) {
                      if (i++) {
                          boost::char_separator<char> sep2(":");
                          boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                          auto it = tok2.begin();
                          int idx = std::stoi(*it++);
                          double val = std::stod(*it++);
                          num_features_agg.update(idx);
                          feature.push_back(val);
                      } else {
                          label = std::stod(w);
                      }
                  }
                  data.add_object(this_obj);
              };
          } else if (this->format_ == kTSVFormat) {
              // TSV format -- [feature]\t[feature]\t[label]
              parser = [&](boost::string_ref chunk) {
                  if (chunk.empty())
                      return;
                  boost::char_separator<char> sep(" \t");
                  boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                  FeatureLabel this_obj;
                  double& label = this_obj.use_label();
                  vec_double& feature = this_obj.use_feature();

                  int i = 0;
                  for (auto& w : tok) {
                      feature.push_back(std::stod(w));
                  }
                  label = feature.back();
                  feature.pop_back();
                  data.add_object(this_obj);
                  num_features_agg.update(feature.size());
              };
          } else {
              ASSERT_MSG(false, "Unsupported data format");
              // parser = [&](boost::string_ref chunk) {};
          }
          husky::load(infmt, {&ac}, parser);
          this->num_feature_ = std::max(this->num_feature_, num_features_agg.get_value());
          // husky::globalize(data);
      }) {
    this->format_ = _format;
}

template <>
DataLoader<SparseFeatureLabel>::DataLoader(int _format)
    : DataLoader([this](std::string url, DataLoader<SparseFeatureLabel>::ObjL& data) {
          husky::io::LineInputFormat infmt;
          infmt.set_input(url);

          husky::lib::Aggregator<int> num_features_agg(0, [](int& a, const int& b) { a = std::max(a, b); });
          auto& ac = husky::lib::AggregatorFactory::get_channel();
          std::function<void(boost::string_ref)> parser;
          if (this->format_ == kLIBSVMFormat) {
              parser = [&](boost::string_ref chunk) {
                  if (chunk.empty())
                      return;
                  boost::char_separator<char> sep(" \t");
                  boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                  SparseFeatureLabel this_obj;
                  double& label = this_obj.use_label();
                  vec_sp& feature = this_obj.use_feature();

                  int i = 0;
                  for (auto& w : tok) {
                      if (i++) {
                          boost::char_separator<char> sep2(":");
                          boost::tokenizer<boost::char_separator<char>> tok2(w, sep2);
                          auto it = tok2.begin();
                          int idx = std::stoi(*it++);
                          double val = std::stod(*it++);
                          num_features_agg.update(idx);
                          feature.push_back(std::make_pair(idx, val));
                      } else {
                          label = std::stod(w);
                      }
                  }
                  data.add_object(this_obj);
              };
          } else if (this->format_ == kTSVFormat) {
              parser = [&](boost::string_ref chunk) {
                  if (chunk.empty())
                      return;
                  boost::char_separator<char> sep(" \t");
                  boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

                  SparseFeatureLabel this_obj;
                  double& label = this_obj.use_label();
                  vec_sp& feature = this_obj.use_feature();

                  int i = 0;
                  for (auto& w : tok) {
                      i++;
                      feature.push_back(std::make_pair(i, std::stod(w)));
                  }
                  label = feature.back().second;
                  feature.pop_back();
                  data.add_object(this_obj);
                  num_features_agg.update(feature.size());
              };
          } else {
              ASSERT_MSG(false, "Unsupported data format");
              // parser = [&](boost::string_ref chunk) {};
          }
          husky::load(infmt, {&ac}, parser);
          this->num_feature_ = std::max(this->num_feature_, num_features_agg.get_value());
      }) {
    this->format_ = _format;
}

}  // namespace ml
}  // namespace lib
}  // namespace husky
