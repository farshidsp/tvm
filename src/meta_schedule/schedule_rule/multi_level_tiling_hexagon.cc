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

using tir::LoopRV;
using tir::Schedule;

/*!
 * \brief TODO
 */
class MultiLevelTilingHexagonNode : public MultiLevelTilingNode {
 protected:
  Array<tir::LoopRV> SplitLoop(Schedule& sch, LoopRV loop, int n_tiles,
                               bool inner_most_spatial) const;

 public:
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingHexagon";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingHexagonNode, MultiLevelTilingNode);
};

Array<tir::LoopRV> MultiLevelTilingHexagonNode::SplitLoop(Schedule& sch, LoopRV loop, int n_tiles,
                                                          bool inner_most_spatial) const {
  const int vec_len = 64;

  if (!inner_most_spatial) {
    return MultiLevelTilingNode::SplitLoop(sch, loop, n_tiles, inner_most_spatial);
  } else {
    const int64_t* extent_int = tir::GetLoopIntExtent(sch->Get(loop).get());
    if (extent_int && *extent_int > vec_len) {
      Array<tir::LoopRV> inner_splits = sch->Split(/*loop=*/loop,
                                                   /*factors=*/{NullOpt, PrimExpr(vec_len)});
      auto inner_loop = inner_splits[0];
      Array<tir::ExprRV> outer_factors = sch->SamplePerfectTile(
          /*loop=*/inner_loop,
          /*n=*/n_tiles - 1,
          /*max_innermost_factor=*/max_innermost_factor);
      Array<tir::LoopRV> outer_splits =
          sch->Split(/*loop=*/inner_loop,
                     /*factors=*/{outer_factors.begin(), outer_factors.end()});
      outer_splits.push_back(inner_splits[1]);
      return outer_splits;
    } else {
      Array<tir::ExprRV> factors(n_tiles - 1, PrimExpr(1));
      factors.push_back(sch->Get(loop)->extent);
      return sch->Split(/*loop=*/loop,
                        /*factors=*/{factors.begin(), factors.end()});
    }
  }
}

ScheduleRule ScheduleRule::MultiLevelTilingHexagon(String structure,
                                                   Optional<Integer> max_innermost_factor,
                                                   Optional<Map<String, ObjectRef>> reuse_read,
                                                   Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingHexagonNode>(
      structure, NullOpt, max_innermost_factor, NullOpt, reuse_read, reuse_write);
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingHexagonNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingHexagon")
    .set_body_typed(ScheduleRule::MultiLevelTilingHexagon);

}  // namespace meta_schedule
}  // namespace tvm
