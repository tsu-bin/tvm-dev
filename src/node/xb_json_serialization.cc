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
 * \file node/serialization.cc
 * \brief Utilities to serialize TVM AST/IR objects.
 */
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/reflection.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cctype>
#include <map>
#include <string>

#include "../runtime/object_internal.h"
#include "../support/base64.h"

#include <sstream>
#define to_hex_str(hex_val) (static_cast<std::stringstream const&>(std::stringstream() << std::hex << hex_val)).str()

namespace tvm {

inline std::string Type2String(const DataType& t) { return runtime::DLDataType2String(t); }

inline DataType String2Type(std::string s) { return DataType(runtime::String2DLDataType(s)); }

inline std::string Base64Decode(std::string s) {
  dmlc::MemoryStringStream mstrm(&s);
  support::Base64InStream b64strm(&mstrm);
  std::string output;
  b64strm.InitPosition();
  dmlc::Stream* strm = &b64strm;
  strm->Read(&output);
  return output;
}

inline std::string Base64Encode(std::string s) {
  std::string blob;
  dmlc::MemoryStringStream mstrm(&blob);
  support::Base64OutStream b64strm(&mstrm);
  dmlc::Stream* strm = &b64strm;
  strm->Write(s);
  b64strm.Finish();
  return blob;
}


class MyJsonPrinter : public AttrVisitor {
  std::ostringstream os_;
  dmlc::JSONWriter writer_;
  std::unordered_set<Object*> visited_;

  bool fold_visited_;

  ReflectionVTable* reflection_ = ReflectionVTable::Global();

public:
  MyJsonPrinter(bool fold_visited) : writer_(&os_), fold_visited_(fold_visited) {
  }

  std::string Print() {
    return os_.str();
  }

  void Visit(const char* key, double* value) final {
    if (not value) return;
    std::ostringstream s;
    // Save 17 decimal digits for type <double> to avoid precision loss during loading JSON
    s.precision(17);
    s << (*value);
    writer_.WriteObjectKeyValue(key, s.str());
  }
  void Visit(const char* key, int64_t* value) final { 
    if (not value) return;
    writer_.WriteObjectKeyValue(key, std::to_string(*value));
  }
  void Visit(const char* key, uint64_t* value) final {
    if (not value) return;
    writer_.WriteObjectKeyValue(key, std::to_string(*value));
  }
  void Visit(const char* key, int* value) final {
    if (not value) return;
    writer_.WriteObjectKeyValue(key, std::to_string(*value));
  }
  void Visit(const char* key, bool* value) final {
    if (not value) return;
    if (*value)
      writer_.WriteObjectKeyValue(key, std::string("True"));
    else
      writer_.WriteObjectKeyValue(key, std::string("False"));
  }
  void Visit(const char* key, std::string* value) final {
    if ((not value) || (value->empty())) return;
    writer_.WriteObjectKeyValue(key, *value);
  }
  void Visit(const char* key, void** value) final {
    if (not value) return;
    //LOG(FATAL) << "not allowed to serialize a pointer";
    writer_.WriteObjectKeyValue(key, std::string("not_support_pointer"));
  }
  void Visit(const char* key, DataType* value) final {
    if (not value) return;
    writer_.WriteObjectKeyValue(key, Type2String(*value));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    if (not value) return;
    //LOG(FATAL) << "not allowed to serialize a runtime::NDArray";
    writer_.WriteObjectKeyValue(key, std::string("not_support_NDArray"));
  }
  
  void Visit(const char* key, ObjectRef* value) final {
    if ((not value) || (not value->get())) return;

    Object* node = const_cast<Object*>(value->get());
    if (node->IsInstance<ArrayNode>()) {
      if (static_cast<ArrayNode*>(node)->size() == 0)
        return;
    } else if (node->IsInstance<MapNode>()) {
      if (static_cast<MapNode*>(node)->size() == 0)
        return;
    }

    std::string repr_bytes;
    if (reflection_->GetReprBytes(node, &repr_bytes) && repr_bytes.empty()) {
      return;
    }

    if (node->GetTypeKey() == "runtime.String") {
      writer_.WriteObjectKeyValue(std::string(key)+" str", repr_bytes);
      return;
    }

    writer_.WriteObjectKey(std::string(key));
    MyVisit(node, key);
  }

  // void Visit(const char* key, runtime::String* value) final {
  //   writer_.WriteObjectKeyValue(std::string(key)+" str", std::string(value->c_str()));
  // }
  
  void MyVisit(Object* node, const char* parent_key = nullptr) {
    if (node == nullptr) {
      writer_.BeginObject();
      writer_.WriteObjectKeyValue("NULLPTR_parent_is", std::string(parent_key?parent_key:"nullptr"));
      writer_.EndObject();
      return;
    }

    if (fold_visited_) {
      if (visited_.count(node)) {
        writer_.BeginObject();
        if (node->IsInstance<ArrayNode>()) {
          writer_.WriteObjectKeyValue("REF_array", to_hex_str((void*)node));
        } else if (node->IsInstance<MapNode>()) {
          writer_.WriteObjectKeyValue("REF_map", to_hex_str((void*)node));
        } else {
          writer_.WriteObjectKeyValue("REF_obj", node->GetTypeKey()+"    "+to_hex_str((void*)node));
        }
        writer_.EndObject();
        return;
      }
      visited_.insert(node);
    }

    writer_.BeginObject();

    if (std::string repr_bytes; reflection_->GetReprBytes(node, &repr_bytes)) {
      writer_.WriteObjectKeyValue("T_A", node->GetTypeKey()+"    "+to_hex_str((void*)node));
      // choose to use str representation or base64, based on whether
      // the byte representation is printable.
      if (std::all_of(repr_bytes.begin(), repr_bytes.end(),
                      [](char ch) { return std::isprint(ch); })) {
        writer_.WriteObjectKeyValue("repr_str", repr_bytes);
      } else {
        writer_.WriteObjectKeyValue("repr_b64", Base64Encode(repr_bytes));
      }
    }
    else if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      int idx = 0;
      for (const auto& sp : *n) {
        std::string kk(std::string(parent_key) + "_E_"+std::to_string(idx));
        writer_.WriteObjectKey(kk);
        MyVisit(const_cast<Object*>(sp.get()));
        idx++;
      }
    } else if (node->IsInstance<MapNode>()) {
      int idx = 0;
      for (const auto& kv : *static_cast<MapNode*>(node)) {
        std::string kk(std::string(parent_key) + "_K_"+std::to_string(idx));
        if (kv.first->IsInstance<StringObj>()) {
          writer_.WriteObjectKeyValue(kk, std::string(Downcast<String>(kv.first)));
        }
        else {
          writer_.WriteObjectKey(kk);
          MyVisit(const_cast<Object*>(kv.first.get()));
        }
        
        writer_.WriteObjectKey(std::string(std::string(parent_key) + "_V_"+std::to_string(idx)));
        MyVisit(const_cast<Object*>(kv.second.get()));
        idx++;
      }
    } 
    else {
      writer_.WriteObjectKeyValue("T_A", node->GetTypeKey()+"    "+to_hex_str((void*)node));
      reflection_->VisitAttrs(const_cast<Object*>(node), this);
    }

    writer_.EndObject();
  }

};

std::string Object::xx() const {
  MyJsonPrinter jp(false);
  jp.MyVisit(const_cast<Object*>(this), "root");

  std::string json_repr = jp.Print();
  //LOG(INFO) << json_repr;
  std::rename("./xb_demo/obj_dump_new.json", "./xb_demo/obj_dump_old.json");

  std::ofstream new_out_file("./xb_demo/obj_dump_new.json");
  new_out_file << json_repr;
  new_out_file.close();

  return json_repr;
}
std::string Object::x() const {
  xx();
  return "Update obj_dump_new.json now!";
}

std::string Object::zz() const {
  MyJsonPrinter jp(true);
  jp.MyVisit(const_cast<Object*>(this), "root");

  std::string json_repr = jp.Print();
  //LOG(INFO) << json_repr;

  std::rename("./xb_demo/obj_dump_new.json", "./xb_demo/obj_dump_old.json");

  std::ofstream new_out_file("./xb_demo/obj_dump_new.json");
  new_out_file << json_repr;
  new_out_file.close();

  return json_repr;
}
std::string Object::z() const {
  zz();
  return "Update obj_dump_new.json now!";
}

std::string ObjectRef::xx() const {
  return this->get()->xx();
}
std::string ObjectRef::x() const {
  return this->get()->x();
}
std::string ObjectRef::zz() const {
  return this->get()->zz();
}
std::string ObjectRef::z() const {
  return this->get()->z();
}


TVM_REGISTER_GLOBAL("runtime.Object_X").set_body_typed([](ObjectRef obj) {
  obj.x();
});
TVM_REGISTER_GLOBAL("runtime.Object_Z").set_body_typed([](ObjectRef obj) {
  obj.z();
});

}  // namespace tvm

std::string x(long obj_ptr) {
  return reinterpret_cast<const tvm::runtime::Object*>(obj_ptr)->x();
}
std::string xx(long obj_ptr) {
  return reinterpret_cast<const tvm::runtime::Object*>(obj_ptr)->xx();
}
std::string z(long obj_ptr) {
  return reinterpret_cast<const tvm::runtime::Object*>(obj_ptr)->z();
}
std::string zz(long obj_ptr) {
  return reinterpret_cast<const tvm::runtime::Object*>(obj_ptr)->zz();
}