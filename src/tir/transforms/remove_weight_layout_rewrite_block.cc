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

/*!
 * \file remove_weight_layout_rewrite_block.cc
 * \brief Remove weight layout rewrite block before benchmark
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

class RemoveLayoutRewriteBlock : public StmtMutator {
 public:
  static std::pair<PrimFunc, Map<Buffer, Buffer>> Rewrite(PrimFunc f) {
    RemoveLayoutRewriteBlock rewriter;
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = rewriter(std::move(n->body));
    return std::make_tuple(f, rewriter.buf_map_);
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));

    auto it = block->annotations.find(tir::attr::meta_schedule_layout_rewrite_preproc);
    if (it == block->annotations.end() || !is_one(Downcast<PrimExpr>((*it).second))) {
      // The block is not a weight layout block
      // Remove allocates if needed
      Array<Buffer> alloc_buffers;
      for (const Buffer& buffer : block->alloc_buffers) {
        if (!rewritten_buffers.count(buffer)) {
          alloc_buffers.push_back(buffer);
        }
      }
      if (alloc_buffers.size() < block->alloc_buffers.size()) {
        auto n = CopyOnWrite(block.get());
        n->alloc_buffers = std::move(alloc_buffers);
        return Stmt(n);
      } else {
        return std::move(block);
      }
    }

    // Step 0. Checking block attrs
    ICHECK(block->alloc_buffers.empty());
    ICHECK(block->match_buffers.empty());

    // Step 1. Checking the body is a BufferStore
    const auto* store = block->body.as<BufferStoreNode>();
    ICHECK(store);

    // Step 2. Checking the rhs of buffer store is a BufferLoad
    const auto* load = store->value.as<BufferLoadNode>();
    ICHECK(load);

    buf_map_.Set(load->buffer, store->buffer);
    rewritten_buffers.insert(store->buffer);

    // Step 4. Set block body as no_op
    auto n = CopyOnWrite(block.get());
    n->body = std::move(Evaluate(0));
    n->reads = {};
    n->writes = {};
    return Stmt(n);
  }

  Map<Buffer, Buffer> buf_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> rewritten_buffers;
};

// HACK
using BufferVarMap = std::unordered_map<const tir::VarNode*, const tir::VarNode*>;

class AllocateConstRewrite : public StmtExprMutator {
 public:
  AllocateConstRewrite(
      const BufferVarMap& buffer_var_map,
      const std::unordered_map<const tir::VarNode*, AllocateConst>& alloc_const_map)
      : buffer_var_map_(buffer_var_map), alloc_const_map_(alloc_const_map) {}

 private:
  Stmt VisitStmt_(const AllocateConstNode* op) final {
    auto body = StmtMutator::VisitStmt(op->body);
    if (auto it = alloc_const_map_.find(op->buffer_var.get()); it != alloc_const_map_.end()) {
      auto rewritten_alloc_const = it->second;
      return AllocateConst(rewritten_alloc_const->buffer_var, rewritten_alloc_const->dtype,
                           rewritten_alloc_const->extents, rewritten_alloc_const->data, body,
                           op->annotations, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (auto it = buffer_var_map_.find(op->buffer->data.get()); it != buffer_var_map_.end()) {
      ICHECK(alloc_const_map_.count(it->second));
      auto rewritten_alloc_const = alloc_const_map_[it->second];
      auto new_buffer =
          Buffer(rewritten_alloc_const->buffer_var, op->buffer->dtype, op->buffer->shape,
                 op->buffer->strides, op->buffer->elem_offset,
                 rewritten_alloc_const->buffer_var->name_hint, op->buffer->data_alignment,
                 op->buffer->offset_factor, op->buffer->buffer_type);
      return BufferLoad(new_buffer, op->indices);
    }
    return ExprMutator::VisitExpr_(op);
  }

  BufferVarMap buffer_var_map_;
  std::unordered_map<const tir::VarNode*, AllocateConst> alloc_const_map_;
};

class WeightLayoutRewriteBlockRemover : public StmtMutator {
 public:
  static PrimFunc Remove(PrimFunc f,
                         std::unordered_map<const tir::VarNode*, AllocateConst> alloc_const_map) {
    auto [f_, buf_map] = RemoveLayoutRewriteBlock().Rewrite(f);

    PrimFuncNode* n = f_.CopyOnWrite();

    if (alloc_const_map.empty()) {
      Map<tir::Var, Buffer> buffer_map;
      for (const auto& [param, buffer] : f->buffer_map) {
        auto it = buf_map.find(buffer);
        if (it != buf_map.end()) {
          buffer_map.Set(param, (*it).second);
        } else {
          buffer_map.Set(param, buffer);
        }
      }
      n->buffer_map = std::move(buffer_map);
      return f_;
    } else {
      BufferVarMap buffer_var_map;
      for (const auto& [load_buf, store_buf] : buf_map) {
        buffer_var_map[store_buf->data.get()] = load_buf->data.get();
      }
      AllocateConstRewrite rewriter(buffer_var_map, alloc_const_map);
      n->body = rewriter(std::move(n->body));
      return f_;
    }
  }
};

namespace transform {

Pass RemoveWeightLayoutRewriteBlock(
    const std::unordered_map<const tir::VarNode*, tir::AllocateConst>& alloc_const_map) {
  auto pass_func = [alloc_const_map](PrimFunc f, IRModule m, PassContext ctx) {
    return WeightLayoutRewriteBlockRemover::Remove(std::move(f), alloc_const_map);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveWeightLayoutRewriteBlock", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveWeightLayoutRewriteBlock")
    .set_body_typed([](Map<tir::Var, tir::AllocateConst> alloc_const_map) {
      std::unordered_map<const tir::VarNode*, tir::AllocateConst> alloc_const_map_;
      for (auto [k, v] : alloc_const_map) {
        alloc_const_map_[k.get()] = v;
      }
      return RemoveWeightLayoutRewriteBlock(alloc_const_map_);
    });

}  // namespace transform

}  // namespace tir
}  // namespace tvm
