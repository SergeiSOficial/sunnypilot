// Read-only inspection of the converted graph; no engine execution.
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include "MNN_generated.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary);
  std::vector<char> data{std::istreambuf_iterator<char>(file), {}};
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  if (!MNN::VerifyNetBuffer(verifier)) return 3;
  const auto* net = MNN::GetNet(data.data());
  for (const auto* op : *net->oplists()) {
    if (op->type() != MNN::OpType_Convolution && op->type() != MNN::OpType_ConvolutionDepthwise) continue;
    const auto* common = op->main_as_Convolution2D()->common();
    std::cout << op->name()->str() << '\t' << MNN::EnumNameOpType(op->type()) << '\t'
              << common->inputCount() << '\t' << common->outputCount() << '\t'
              << common->group() << '\t' << common->kernelY() << '\t' << common->kernelX() << '\n';
  }
}
