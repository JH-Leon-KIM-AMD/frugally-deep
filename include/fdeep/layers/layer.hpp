// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/tensor.hpp"

#include "fdeep/node.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class layer;
    typedef std::shared_ptr<layer> layer_ptr;
    typedef std::vector<layer_ptr> layer_ptrs;

    class activation_layer;
    typedef std::shared_ptr<activation_layer> activation_layer_ptr;
    tensors apply_activation_layer(const activation_layer_ptr& ptr,
        const tensors& input);

    class layer {
    public:
        explicit layer(const std::string& name)
            : name_(name)
            , nodes_()
            , activation_(nullptr)
        {
        }
        virtual ~layer()
        {
        }

        void set_activation(const activation_layer_ptr& activation)
        {
            activation_ = activation;
        }

        void set_nodes(const nodes& layer_nodes)
        {
            nodes_ = layer_nodes;
        }

        virtual tensors apply(const tensors& input) const final
        {
            const auto result = apply_impl(input);
            if (activation_ == nullptr)
                return result;
            else
                return apply_activation_layer(activation_, result);
        }

        virtual tensor get_output(const layer_ptrs& layers,
            output_dict& output_cache,
            std::size_t node_idx, std::size_t tensor_idx) const
        {
            auto key = std::make_pair(name_, node_idx);
            auto it = output_cache.find(key);
            if (it == output_cache.end()) {
                assertion(node_idx < nodes_.size(), "invalid node index");
                auto outputs = nodes_[node_idx].get_output(layers, output_cache, *this);
                it = output_cache.emplace(std::move(key), std::move(outputs)).first;
            }

            assertion(tensor_idx < it->second.size(),
                "invalid tensor index");
            return it->second[tensor_idx];
        }

        virtual void reset_states()
        {
            // Stateful layers should override that function,
            // and take care to reset their internal states if appropriate.
        }

        virtual bool is_stateful() const
        {
            return false;
            // Stateful layers should override that function, with return true.
        }

        std::string name_;
        nodes nodes_;

    protected:
        virtual tensors apply_impl(const tensors& input) const = 0;
        activation_layer_ptr activation_;
    };

    inline tensor get_layer_output(const layer_ptrs& layers,
        output_dict& output_cache,
        const layer_ptr& layer,
        std::size_t node_idx, std::size_t tensor_idx)
    {
        return layer->get_output(layers, output_cache, node_idx, tensor_idx);
    }

    inline tensors apply_layer(const layer& layer, const tensors& inputs)
    {
        return layer.apply(inputs);
    }

    inline const layer_ptr& get_layer(const layer_ptrs& layers,
        const std::string& layer_id)
    {
        for (const auto& ptr : layers) {
            if (ptr->name_ == layer_id)
                return ptr;
        }
        throw error("dangling layer reference: " + layer_id);
    }

}
}
