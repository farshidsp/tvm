/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "../../tir/schedule/analysis.h"
#include "../../tir/schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingHexagonNode : public MultiLevelTilingNode {
 protected:
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    return MultiLevelTilingNode::Apply(sch->Copy(), block_rv);
  }

  // Override ApplySubRules to tile the inner loops according to the given tensor intrinsic, then
  // tile the outerloops.
  virtual std::vector<State> ApplySubRules(std::vector<State> states) {
    return MultiLevelTilingNode::ApplySubRules(states);
  }

 public:
  /*! \brief The name of a tensor intrinsic. */
  String intrin_name;

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingHexagon";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingHexagonNode, MultiLevelTilingNode);
};

ScheduleRule ScheduleRule::MultiLevelTilingHexagon(
    String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingHexagonNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingHexagonNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingHexagon")
    .set_body_typed(ScheduleRule::MultiLevelTilingHexagon);

}  // namespace meta_schedule
}  // namespace tvm
